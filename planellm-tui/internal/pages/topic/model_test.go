package topic

import (
	"errors"
	"strings"
	"testing"
	"time"

	tea "charm.land/bubbletea/v2"
	"charm.land/huh/v2"
	ctx "github.com/jasperan/planellm-tui/internal/context"
	"github.com/jasperan/planellm-tui/internal/theme"
)

// This page previously had no tests. The surface it exposes is a form, and a form
// fails in ways a compiler cannot see: a key that stops matching still compiles,
// and a field that stops receiving text still renders. So these tests press keys
// and assert the EFFECT, in the same style as the menu and shell suites.

// stubAPI records what the page asked the backend to do. The real client needs a
// reachable planeLLM service, which a unit test must not require.
type stubAPI struct {
	topicCalls []string
	demoCalls  []string
	result     *ctx.TopicResult
	err        error
}

func (s *stubAPI) GetStatus() (*ctx.SystemStatus, error) { return &ctx.SystemStatus{}, nil }
func (s *stubAPI) ListFiles() (*ctx.FileList, error)     { return &ctx.FileList{}, nil }

func (s *stubAPI) GenerateTopic(topic string) (*ctx.TopicResult, error) {
	s.topicCalls = append(s.topicCalls, topic)
	return s.result, s.err
}

func (s *stubAPI) BootstrapDemo(topic string) (*ctx.TopicResult, error) {
	s.demoCalls = append(s.demoCalls, topic)
	return s.result, s.err
}

func (s *stubAPI) CreateTranscript(string, bool) (*ctx.TranscriptResult, error) {
	return &ctx.TranscriptResult{}, nil
}

func (s *stubAPI) GenerateAudio(string, string, string, string) (*ctx.AudioResult, error) {
	return &ctx.AudioResult{}, nil
}

func newTestModel(t *testing.T, api ctx.APIClient) *Model {
	t.Helper()
	m := New(&ctx.Context{Theme: theme.New(), API: api})
	// Init must run before any keystroke: huh activates the first group and
	// focuses its first field there, and a field that was never focused receives
	// nothing. The shell calls Init, so the test has to as well.
	pump(t, m, m.Init()(), nil)
	m.Update(tea.WindowSizeMsg{Width: 120, Height: 40})
	return m
}

// namedKey builds the message the runtime delivers for a named key and asserts it
// really reports that name. This is the guard the menu suite uses, and it is what
// caught the v1/v2 " " vs "space" rename.
func namedKey(t *testing.T, name string, k tea.KeyPressMsg) tea.KeyPressMsg {
	t.Helper()
	if got := k.String(); got != name {
		t.Fatalf("key construction: message reports %q but the test asked for %q", got, name)
	}
	return k
}

func spaceKey(t *testing.T) tea.KeyPressMsg {
	return namedKey(t, "space", tea.KeyPressMsg{Code: ' ', Text: " "})
}

func runeKey(t *testing.T, r rune) tea.KeyPressMsg {
	return namedKey(t, string(r), tea.KeyPressMsg{Code: r, Text: string(r)})
}

func enterKey(t *testing.T) tea.KeyPressMsg {
	return namedKey(t, "enter", tea.KeyPressMsg{Code: tea.KeyEnter})
}

func escKey(t *testing.T) tea.KeyPressMsg {
	return namedKey(t, "esc", tea.KeyPressMsg{Code: tea.KeyEsc})
}

// cmdBound bounds a single command run. This is not optional: huh re-arms the
// text input cursor blink on every update and each blink tick sleeps about 530ms,
// so an unbounded drain either hangs the suite or takes minutes. A sibling agent in
// this program lost half an hour to exactly that.
//
// A command that does not answer within the bound is abandoned and treated as a
// sleeping tick (blink, spinner). Only this page's own closures matter to the
// assertions, and those return immediately — so a starved machine makes these tests
// fail loudly rather than pass on a dropped message.
const cmdBound = 25 * time.Millisecond

// maxPump bounds the feedback loop, so a command that keeps re-emitting itself
// (a blink or spinner tick) cannot make a test spin forever.
const maxPump = 64

// runBounded runs one command tree and returns every leaf message it produced,
// abandoning any command that does not answer within cmdBound.
func runBounded(t *testing.T, cmd tea.Cmd) []tea.Msg {
	t.Helper()
	if cmd == nil {
		return nil
	}

	done := make(chan tea.Msg, 1)
	go func() {
		defer func() { recover() }()
		done <- cmd()
	}()

	var msg tea.Msg
	select {
	case msg = <-done:
	case <-time.After(cmdBound):
		return nil
	}

	if batch, ok := msg.(tea.BatchMsg); ok {
		var out []tea.Msg
		for _, sub := range batch {
			out = append(out, runBounded(t, sub)...)
		}
		return out
	}
	return []tea.Msg{msg}
}

// pump drives one message through the model and then feeds every command result
// back in until the model stops producing commands.
//
// Feeding the results back is the whole point of this harness. huh reports progress
// through *commands*, not through return values: the input returns a command
// yielding nextFieldMsg, the group turns that into nextGroupMsg, and only the form
// turns that into a completion. A harness that merely runs the command and throws
// its message away never completes a form — which is exactly how the first version
// of these tests failed.
//
// observe, when non-nil, runs after every Update so a test can capture transient
// state (such as "loading, with the field still on screen") that is gone by the
// time the loop settles.
func pump(t *testing.T, m *Model, initial tea.Msg, observe func(*Model)) []tea.Msg {
	t.Helper()

	var seen []tea.Msg
	var pending []tea.Msg
	if initial != nil {
		pending = append(pending, initial)
	}

	for i := 0; i < maxPump && len(pending) > 0; i++ {
		msg := pending[0]
		pending = pending[1:]
		seen = append(seen, msg)

		_, cmd := m.Update(msg)
		if observe != nil {
			observe(m)
		}
		pending = append(pending, runBounded(t, cmd)...)
	}
	return seen
}

// press drives one key to quiescence.
func press(t *testing.T, m *Model, k tea.KeyPressMsg) []tea.Msg {
	t.Helper()
	return pump(t, m, k, nil)
}

// typeText feeds a string to the focused field one rune at a time, exactly as a
// terminal would. Space is exercised explicitly: it is the character most likely to
// be swallowed by a key binding, and this module has already lost a binding to the
// v2 rename of the space bar.
//
// The command each keystroke returns is deliberately not run. For a printable rune
// that command is only the text input's cursor blink, which carries nothing the
// assertions read and which the next update re-arms anyway — running it would just
// make every keystroke wait out cmdBound. Submit keys go through press, which does
// pump, because there the chained messages are the behaviour under test.
func typeText(t *testing.T, m *Model, s string) {
	t.Helper()
	for _, r := range s {
		var k tea.KeyPressMsg
		if r == ' ' {
			k = spaceKey(t)
		} else {
			k = runeKey(t, r)
		}
		m.Update(k)
	}
}

// topicResultOf finds the page's own result message in a pumped transcript.
func topicResultOf(t *testing.T, msgs []tea.Msg) *topicResultMsg {
	t.Helper()
	for _, msg := range msgs {
		if r, ok := msg.(topicResultMsg); ok {
			return &r
		}
	}
	return nil
}

func TestValidateTopicRejectsBlankAnswers(t *testing.T) {
	for _, in := range []string{"", " ", "\t", "   \n  "} {
		if err := validateTopic(in); err == nil {
			t.Errorf("validateTopic(%q) = nil, want an error — a blank topic cannot generate anything", in)
		}
	}
}

func TestValidateTopicAcceptsRealTopics(t *testing.T) {
	for _, in := range []string{"black holes", "  quantum computing  ", "a"} {
		if err := validateTopic(in); err != nil {
			t.Errorf("validateTopic(%q) = %v, want nil", in, err)
		}
	}
}

// The regression this file most needs to hold: the space bar. v1 reported it as
// " " and v2 reports "space", and this module has already shipped a binding that
// silently stopped matching because of that rename. A form field that lost spaces
// would be far subtler than a dead key — the text would still appear, just with the
// spaces gone.
func TestTypingASpaceReachesTheBoundField(t *testing.T) {
	m := newTestModel(t, &stubAPI{})

	typeText(t, m, "black holes")

	if m.topic != "black holes" {
		t.Fatalf("bound topic = %q, want %q — the space keystroke did not reach the form", m.topic, "black holes")
	}
}

func TestSpaceIsNotConsumedByThePageShortcuts(t *testing.T) {
	m := newTestModel(t, &stubAPI{})

	press(t, m, spaceKey(t))
	press(t, m, runeKey(t, 'a'))

	if m.topic != " a" {
		t.Errorf("bound topic = %q, want %q — the page consumed the leading space", m.topic, " a")
	}
	if m.loading {
		t.Error("a space started a load; space must only ever be text here")
	}
}

func TestEnterOnABlankFieldShowsAnErrorInsteadOfDoingNothing(t *testing.T) {
	api := &stubAPI{}
	m := newTestModel(t, api)

	// The old hand-rolled input ignored Enter on an empty field entirely, which
	// reads as a broken key. Now the form refuses and says why.
	press(t, m, spaceKey(t))
	press(t, m, enterKey(t))

	if m.loading {
		t.Error("Enter on a blank field started a load")
	}
	if len(api.topicCalls) != 0 {
		t.Errorf("Enter on a blank field called the backend %d times", len(api.topicCalls))
	}
	if !strings.Contains(m.View(), "enter a topic") {
		t.Errorf("no validation message was rendered for a blank answer:\n%s", m.View())
	}
}

func TestEnterSubmitsTheTypedTopicToTheBackend(t *testing.T) {
	api := &stubAPI{result: &ctx.TopicResult{Success: true, Questions: []string{"q1"}}}
	m := newTestModel(t, api)

	sawLoading := false
	typeText(t, m, "black holes")
	msgs := pump(t, m, enterKey(t), func(m *Model) {
		if m.loading {
			sawLoading = true
		}
	})

	if !sawLoading {
		t.Error("Enter never put the page into its loading state")
	}
	if len(api.topicCalls) != 1 || api.topicCalls[0] != "black holes" {
		t.Errorf("backend calls = %v, want exactly one call with %q", api.topicCalls, "black holes")
	}
	if result := topicResultOf(t, msgs); result == nil {
		t.Error("no topicResultMsg was produced — the submit did not reach generateTopic")
	}
	// The result the backend returned must have landed on the page.
	if m.result == nil || !m.result.Success {
		t.Errorf("result = %+v, want the successful result from the backend", m.result)
	}
}

func TestCtrlDStartsTheDemoBundle(t *testing.T) {
	api := &stubAPI{result: &ctx.TopicResult{Success: true}}
	m := newTestModel(t, api)

	sawLoading := false
	msgs := pump(t, m, namedKey(t, "ctrl+d", tea.KeyPressMsg{Code: 'd', Mod: tea.ModCtrl}), func(m *Model) {
		if m.loading {
			sawLoading = true
		}
	})

	if !sawLoading {
		t.Error("ctrl+d never put the page into its loading state")
	}
	if len(api.demoCalls) != 1 {
		t.Fatalf("demo calls = %v, want exactly 1", api.demoCalls)
	}
	if len(api.topicCalls) != 0 {
		t.Errorf("ctrl+d also called GenerateTopic: %v", api.topicCalls)
	}
	if result := topicResultOf(t, msgs); result == nil {
		t.Error("ctrl+d produced no topicResultMsg")
	}
}

// The demo path is documented as usable with an empty field, and it substitutes a
// default topic rather than sending nothing.
func TestCtrlDSendsADefaultTopicWhenTheFieldIsEmpty(t *testing.T) {
	api := &stubAPI{result: &ctx.TopicResult{Success: true}}
	m := newTestModel(t, api)

	press(t, m, namedKey(t, "ctrl+d", tea.KeyPressMsg{Code: 'd', Mod: tea.ModCtrl}))

	if len(api.demoCalls) != 1 {
		t.Fatalf("demo calls = %v, want exactly 1", api.demoCalls)
	}
	if strings.TrimSpace(api.demoCalls[0]) == "" {
		t.Error("the demo was sent an empty topic; the fallback topic was not applied")
	}
}

func TestCtrlRClearsTheAnswerAndTheResult(t *testing.T) {
	api := &stubAPI{result: &ctx.TopicResult{Success: true}}
	m := newTestModel(t, api)

	typeText(t, m, "black holes")

	press(t, m, namedKey(t, "ctrl+r", tea.KeyPressMsg{Code: 'r', Mod: tea.ModCtrl}))

	if m.topic != "" {
		t.Errorf("after reset topic = %q, want empty", m.topic)
	}
	if m.result != nil {
		t.Error("after reset the previous result was still set")
	}
	if m.loading {
		t.Error("after reset the page was still loading")
	}
	if !strings.Contains(m.View(), "Podcast topic") {
		t.Errorf("after reset the field was gone:\n%s", m.View())
	}
}

// A huh form that has been submitted sets an internal quitting flag, after which
// View renders an empty string forever. The shell keeps one Model per page and calls
// Init() again whenever NavigateMsg re-selects the page, so an un-re-armed form would
// show a blank screen on the second visit with no field to type into.
//
// This test is also what proves the re-arm exists: with the re-arm removed it is the
// second Visit() assertion that fails.
func TestPageRendersAFieldOnEveryVisit(t *testing.T) {
	api := &stubAPI{result: &ctx.TopicResult{Success: true}}
	m := newTestModel(t, api)

	if !strings.Contains(m.View(), "Podcast topic") {
		t.Fatalf("first visit did not render the field:\n%s", m.View())
	}

	typeText(t, m, "black holes")
	press(t, m, enterKey(t))

	if !strings.Contains(m.View(), "Podcast topic") {
		t.Fatalf("the field disappeared after submitting (a spent huh form renders nothing):\n%s", m.View())
	}

	// Re-entering the page is exactly what the shell does: call Init() on the
	// retained model.
	m.Update(tea.WindowSizeMsg{Width: 120, Height: 40})
	pump(t, m, m.Init()(), nil)

	if !strings.Contains(m.View(), "Podcast topic") {
		t.Fatalf("second visit rendered no field (blank page):\n%s", m.View())
	}
	if m.topic != "black holes" {
		t.Errorf("the typed topic was lost across a re-arm: %q", m.topic)
	}
}

// A submitted form must accept a new answer: Enter has to work again, which the old
// always-visible input allowed.
func TestASecondTopicCanBeSubmittedAfterTheFirst(t *testing.T) {
	api := &stubAPI{result: &ctx.TopicResult{Success: true}}
	m := newTestModel(t, api)

	typeText(t, m, "first")
	press(t, m, enterKey(t))

	press(t, m, namedKey(t, "ctrl+r", tea.KeyPressMsg{Code: 'r', Mod: tea.ModCtrl}))
	typeText(t, m, "second")
	press(t, m, enterKey(t))

	want := []string{"first", "second"}
	if len(api.topicCalls) != len(want) {
		t.Fatalf("backend calls = %v, want %v", api.topicCalls, want)
	}
	for i := range want {
		if api.topicCalls[i] != want[i] {
			t.Errorf("call %d = %q, want %q", i, api.topicCalls[i], want[i])
		}
	}
}

// The field stays on screen while the backend works, which is what the old
// always-visible input did.
func TestFieldStaysVisibleWhileLoading(t *testing.T) {
	api := &stubAPI{result: &ctx.TopicResult{Success: true}}
	m := newTestModel(t, api)

	loadingWithField := false
	resultArrivedWhileLoading := false
	typeText(t, m, "black holes")
	pump(t, m, enterKey(t), func(m *Model) {
		if !m.loading {
			return
		}
		if strings.Contains(m.View(), "Podcast topic") {
			loadingWithField = true
		}
		if m.result != nil || m.err != nil {
			resultArrivedWhileLoading = true
		}
	})

	if !loadingWithField {
		t.Error("the field was not on screen during the load")
	}
	if resultArrivedWhileLoading {
		t.Error("setup problem: the backend answered before the loading state was observed")
	}
}

func TestResultIsRenderedOnceTheBackendAnswers(t *testing.T) {
	api := &stubAPI{result: &ctx.TopicResult{
		Success:       true,
		QuestionsFile: "/tmp/questions.md",
		Questions:     []string{"Why?"},
	}}
	m := newTestModel(t, api)

	typeText(t, m, "black holes")
	press(t, m, enterKey(t))

	if m.loading {
		t.Error("the page was still loading after the backend answered")
	}
	view := m.View()
	if !strings.Contains(view, "/tmp/questions.md") {
		t.Errorf("the result path was not rendered:\n%s", view)
	}
	if !strings.Contains(view, "Why?") {
		t.Errorf("the generated questions were not rendered:\n%s", view)
	}
}

func TestBackendFailureIsSurfacedAndTheAnswerIsKept(t *testing.T) {
	api := &stubAPI{err: errors.New("service unavailable")}
	m := newTestModel(t, api)

	typeText(t, m, "black holes")
	press(t, m, enterKey(t))

	if m.err == nil {
		t.Fatal("the backend failure was not stored on the page")
	}
	if !strings.Contains(m.View(), "Error:") {
		t.Errorf("the error was not rendered:\n%s", m.View())
	}
	// A failure must not throw away the user's answer, or they have to retype it.
	if m.topic != "black holes" {
		t.Errorf("the typed topic was discarded on failure: %q", m.topic)
	}
}

func TestUnsuccessfulResultIsReportedAsAnError(t *testing.T) {
	m := newTestModel(t, &stubAPI{})

	pump(t, m, topicResultMsg{result: &ctx.TopicResult{Success: false, Message: "no model configured"}}, nil)

	if m.err == nil {
		t.Fatal("an unsuccessful result was not turned into an error")
	}
	if !strings.Contains(m.err.Error(), "no model configured") {
		t.Errorf("error = %q, want it to carry the backend message", m.err)
	}
}

func TestMissingAPIClientIsReportedRatherThanPanicking(t *testing.T) {
	m := New(&ctx.Context{Theme: theme.New()})
	pump(t, m, m.Init()(), nil)
	m.Update(tea.WindowSizeMsg{Width: 120, Height: 40})

	typeText(t, m, "black holes")
	press(t, m, enterKey(t))

	if m.err == nil {
		t.Fatal("a nil API client was not reported as an error")
	}
}

// Escape belongs to the shell, which returns to the menu before any page sees the
// key. That is why this page has no StateAborted branch: huh's abort binding can
// never fire here, so adding one would be unreachable code. This test pins the fact
// that reasoning depends on — esc handed straight to the page is inert.
func TestEscapeDeliveredToThePageIsInert(t *testing.T) {
	m := newTestModel(t, &stubAPI{})

	typeText(t, m, "black holes")
	press(t, m, escKey(t))

	if m.form.State != huh.StateNormal {
		t.Errorf("esc moved the form out of StateNormal (%v); huh's abort binding is now reachable and this page needs a branch for it", m.form.State)
	}
	if m.topic != "black holes" {
		t.Errorf("esc changed the answer to %q", m.topic)
	}
}

// Re-arming must replay the terminal size into the fresh form.
//
// A new form has no height, and huh only derives one from a size message, so a
// re-armed form that is never re-sized renders CLIPPED: the title and description
// survive but the field row and the bottom border are cut off. A "not blank"
// assertion cannot see that, which is why this test asserts the structure instead.
func TestReArmKeepsTheFieldRowAndBottomBorder(t *testing.T) {
	api := &stubAPI{result: &ctx.TopicResult{Success: true}}
	m := newTestModel(t, api)

	typeText(t, m, "black holes")
	press(t, m, enterKey(t))

	view := m.View()
	if !strings.Contains(view, "black holes") {
		t.Errorf("the field row is gone after a re-arm (the form was not re-sized):\n%s", view)
	}
	if !strings.Contains(view, "╰") {
		t.Errorf("the form's bottom border is gone after a re-arm (the form was not re-sized):\n%s", view)
	}

	// And the same after an explicit re-entry, which is the path the shell takes.
	m.Update(tea.WindowSizeMsg{Width: 120, Height: 40})
	pump(t, m, m.Init()(), nil)

	view = m.View()
	if !strings.Contains(view, "black holes") || !strings.Contains(view, "╰") {
		t.Errorf("re-entry rendered a clipped form:\n%s", view)
	}
}
