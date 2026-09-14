package menu

import (
	"testing"

	tea "charm.land/bubbletea/v2"
	"github.com/jasperan/planellm-tui/internal/app"
	ctx "github.com/jasperan/planellm-tui/internal/context"
	"github.com/jasperan/planellm-tui/internal/theme"
)

// This module had no tests at all before the v2 migration. The migration is
// mostly compiler-invisible — a `case "x":` body that stops matching still
// compiles, still vets, and still passes a test that only asserts the keymap —
// so these tests press keys and assert the EFFECT.

func newTestModel() *Model {
	return New(&ctx.Context{Theme: theme.New()})
}

// namedKey builds the message the runtime delivers for a named key, and asserts
// it really reports that name.
//
// This self-check matters: without it a test could construct a message whose
// String() is something else entirely, assert an effect that came from the
// wrong branch, or pass while exercising nothing. It is also the guard that
// caught " " vs "space".
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

// update drives one key and returns the resulting message from the command, if any.
func update(t *testing.T, m *Model, k tea.KeyPressMsg) tea.Msg {
	t.Helper()
	_, cmd := m.Update(k)
	if cmd == nil {
		return nil
	}
	return cmd()
}

func navigateTarget(t *testing.T, m *Model, k tea.KeyPressMsg) (app.Page, bool) {
	t.Helper()
	msg := update(t, m, k)
	nav, ok := msg.(app.NavigateMsg)
	return nav.Page, ok
}

// The regression this file exists for: v1 named the space bar " " and v2 names it
// "space". The original binding was `case "enter", " "`, which compiled and never
// matched, so Space silently stopped selecting.
func TestSpaceSelectsTheHighlightedPage(t *testing.T) {
	m := newTestModel()

	page, ok := navigateTarget(t, m, spaceKey(t))
	if !ok {
		t.Fatal("Space produced no NavigateMsg — the binding is not matching (v2 reports the space bar as \"space\", not \" \")")
	}
	if want := items[0].page; page != want {
		t.Errorf("Space navigated to %v, want %v", page, want)
	}
}

func TestEnterSelectsTheHighlightedPage(t *testing.T) {
	m := newTestModel()

	page, ok := navigateTarget(t, m, namedKey(t, "enter", tea.KeyPressMsg{Code: tea.KeyEnter}))
	if !ok {
		t.Fatal("Enter produced no NavigateMsg")
	}
	if want := items[0].page; page != want {
		t.Errorf("Enter navigated to %v, want %v", page, want)
	}
}

// Space must select whatever the cursor is on — not merely produce a message. If
// the binding were matching some other key that happened to fall through to the
// same branch, this would still catch a wrong target.
func TestSpaceSelectsWhateverIsUnderTheCursor(t *testing.T) {
	for i := range items {
		m := newTestModel()
		for n := 0; n < i; n++ {
			update(t, m, namedKey(t, "down", tea.KeyPressMsg{Code: tea.KeyDown}))
		}
		if m.cursor != i {
			t.Fatalf("setup: cursor = %d, want %d", m.cursor, i)
		}

		page, ok := navigateTarget(t, m, spaceKey(t))
		if !ok {
			t.Fatalf("row %d: Space produced no NavigateMsg", i)
		}
		if page != items[i].page {
			t.Errorf("row %d: Space navigated to %v, want %v", i, page, items[i].page)
		}
	}
}

// Space and Enter are documented as the same action, so they must agree on every row.
func TestSpaceAndEnterAgreeOnEveryRow(t *testing.T) {
	for i := range items {
		spaceModel := newTestModel()
		enterModel := newTestModel()
		for n := 0; n < i; n++ {
			update(t, spaceModel, namedKey(t, "down", tea.KeyPressMsg{Code: tea.KeyDown}))
			update(t, enterModel, namedKey(t, "down", tea.KeyPressMsg{Code: tea.KeyDown}))
		}

		viaSpace, okSpace := navigateTarget(t, spaceModel, spaceKey(t))
		viaEnter, okEnter := navigateTarget(t, enterModel, namedKey(t, "enter", tea.KeyPressMsg{Code: tea.KeyEnter}))
		if !okSpace || !okEnter {
			t.Fatalf("row %d: space ok=%v enter ok=%v, want both true", i, okSpace, okEnter)
		}
		if viaSpace != viaEnter {
			t.Errorf("row %d: Space goes to %v but Enter goes to %v", i, viaSpace, viaEnter)
		}
	}
}

func TestCursorKeysMoveTheSelection(t *testing.T) {
	down := []struct {
		name string
		key  tea.KeyPressMsg
	}{
		{"down", namedKey(t, "down", tea.KeyPressMsg{Code: tea.KeyDown})},
		{"j", runeKey(t, 'j')},
	}
	up := []struct {
		name string
		key  tea.KeyPressMsg
	}{
		{"up", namedKey(t, "up", tea.KeyPressMsg{Code: tea.KeyUp})},
		{"k", runeKey(t, 'k')},
	}

	for _, k := range down {
		m := newTestModel()
		update(t, m, k.key)
		if m.cursor != 1 {
			t.Errorf("%s: cursor = %d, want 1", k.name, m.cursor)
		}
	}
	for _, k := range up {
		m := newTestModel()
		update(t, m, namedKey(t, "down", tea.KeyPressMsg{Code: tea.KeyDown}))
		update(t, m, k.key)
		if m.cursor != 0 {
			t.Errorf("%s: cursor = %d, want 0", k.name, m.cursor)
		}
	}
}

func TestCursorStopsAtBothEnds(t *testing.T) {
	m := newTestModel()
	update(t, m, namedKey(t, "up", tea.KeyPressMsg{Code: tea.KeyUp}))
	if m.cursor != 0 {
		t.Errorf("up at the top moved the cursor to %d, want 0", m.cursor)
	}

	for i := 0; i < len(items)+3; i++ {
		update(t, m, namedKey(t, "down", tea.KeyPressMsg{Code: tea.KeyDown}))
	}
	if want := len(items) - 1; m.cursor != want {
		t.Errorf("down past the end left the cursor at %d, want %d", m.cursor, want)
	}
}

func TestQuitKeysReturnQuit(t *testing.T) {
	for _, k := range []struct {
		name string
		key  tea.KeyPressMsg
	}{
		{"q", runeKey(t, 'q')},
		{"esc", namedKey(t, "esc", tea.KeyPressMsg{Code: tea.KeyEsc})},
	} {
		m := newTestModel()
		if _, ok := update(t, m, k.key).(tea.QuitMsg); !ok {
			t.Errorf("%s did not produce tea.QuitMsg", k.name)
		}
	}
}

// Every key the menu handles must produce an observable effect. This is the
// general form of the Space regression: a case body that stops matching yields
// no effect at all, and only a press-and-observe test can see that.
func TestEveryHandledKeyHasAnObservableEffect(t *testing.T) {
	all := []struct {
		name string
		key  tea.KeyPressMsg
	}{
		{"q", runeKey(t, 'q')},
		{"esc", namedKey(t, "esc", tea.KeyPressMsg{Code: tea.KeyEsc})},
		{"up", namedKey(t, "up", tea.KeyPressMsg{Code: tea.KeyUp})},
		{"k", runeKey(t, 'k')},
		{"down", namedKey(t, "down", tea.KeyPressMsg{Code: tea.KeyDown})},
		{"j", runeKey(t, 'j')},
		{"enter", namedKey(t, "enter", tea.KeyPressMsg{Code: tea.KeyEnter})},
		{"space", spaceKey(t)},
	}

	for _, k := range all {
		// Start from the middle row so up, down, k and j all have somewhere to go.
		m := newTestModel()
		update(t, m, namedKey(t, "down", tea.KeyPressMsg{Code: tea.KeyDown}))
		before := m.cursor

		msg := update(t, m, k.key)

		moved := m.cursor != before
		_, quit := msg.(tea.QuitMsg)
		_, navigated := msg.(app.NavigateMsg)

		if !moved && !quit && !navigated {
			t.Errorf("key %q produced no observable effect (cursor %d -> %d, msg %T) — "+
				"the case body is not matching", k.name, before, m.cursor, msg)
		}
	}
}

// In v2 tea.KeyMsg is an INTERFACE satisfied by both KeyPressMsg and KeyReleaseMsg, and
// a release reports the same name as its press:
//
//	tea.KeyReleaseMsg{Code: ' ', Text: " "}.String() == "space"
//
// So a bare `case tea.KeyMsg:` — the v1 form — matches releases as well as presses, and
// on a terminal that reports key releases (the Kitty keyboard protocol) every binding
// would fire on release too, i.e. twice per keystroke. Matching tea.KeyPressMsg
// explicitly is what prevents that, and these tests are what make the distinction
// load-bearing: without them, reverting to `case tea.KeyMsg:` fails nothing.
func TestKeyReleaseDoesNotTriggerTheAction(t *testing.T) {
	releases := []struct {
		name string
		key  tea.KeyReleaseMsg
	}{
		{"space", tea.KeyReleaseMsg{Code: ' ', Text: " "}},
		{"enter", tea.KeyReleaseMsg{Code: tea.KeyEnter}},
		{"down", tea.KeyReleaseMsg{Code: tea.KeyDown}},
		{"up", tea.KeyReleaseMsg{Code: tea.KeyUp}},
		{"j", tea.KeyReleaseMsg{Code: 'j', Text: "j"}},
		{"esc", tea.KeyReleaseMsg{Code: tea.KeyEsc}},
		{"q", tea.KeyReleaseMsg{Code: 'q', Text: "q"}},
	}

	for _, r := range releases {
		m := newTestModel()
		update(t, m, namedKey(t, "down", tea.KeyPressMsg{Code: tea.KeyDown}))
		before := m.cursor

		_, cmd := m.Update(r.key)

		var msg tea.Msg
		if cmd != nil {
			msg = cmd()
		}
		if moved := m.cursor != before; moved {
			t.Errorf("releasing %q moved the cursor %d -> %d; a release must not act", r.name, before, m.cursor)
		}
		if _, navigated := msg.(app.NavigateMsg); navigated {
			t.Errorf("releasing %q navigated; a release must not act", r.name)
		}
		if _, quit := msg.(tea.QuitMsg); quit {
			t.Errorf("releasing %q quit; a release must not act", r.name)
		}
	}
}
