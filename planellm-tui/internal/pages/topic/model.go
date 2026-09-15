package topic

import (
	"errors"
	"fmt"
	"strings"

	"charm.land/bubbles/v2/spinner"
	tea "charm.land/bubbletea/v2"
	"charm.land/huh/v2"
	"charm.land/lipgloss/v2"
	"github.com/jasperan/planellm-tui/internal/app"
	ctx "github.com/jasperan/planellm-tui/internal/context"
	"github.com/jasperan/planellm-tui/internal/huhstyle"
)

// topicCharLimit is the ceiling the hand-rolled textinput enforced. The backend
// rejects longer topics, so this is a correctness bound rather than styling.
const topicCharLimit = 200

// formGutter is the two-space page margin on each side of the form.
const formGutter = 4

// minFormWidth keeps a very narrow terminal from collapsing the field to nothing.
const minFormWidth = 20

// formBlock applies the page's two-space left gutter to every line of the form.
var formBlock = lipgloss.NewStyle().PaddingLeft(2)

// fieldTopic keys the form field so tests can read the answer back by name
// instead of reaching into huh internals.
const fieldTopic = "topic"

type topicResultMsg struct {
	result *ctx.TopicResult
	err    error
}

type Model struct {
	ctx     *ctx.Context
	form    *huh.Form
	topic   string
	spinner spinner.Model
	result  *ctx.TopicResult
	loading bool
	err     error
	width   int
	height  int
}

// validateTopic replaces a silent failure. The old hand-rolled input gated Enter
// on `m.input.Value() != ""` and otherwise did nothing at all, which reads as a
// broken key rather than as a missing answer.
func validateTopic(value string) error {
	if strings.TrimSpace(value) == "" {
		return errors.New("enter a topic to generate content for")
	}
	return nil
}

func New(c *ctx.Context) *Model {
	s := spinner.New()
	s.Spinner = spinner.Dot
	s.Style = lipgloss.NewStyle().Foreground(c.Theme.Accent)

	m := &Model{
		ctx:     c,
		spinner: s,
	}
	m.form = m.newForm()
	return m
}

func (m *Model) newForm() *huh.Form {
	f := huh.NewForm(
		huh.NewGroup(
			huh.NewInput().
				Key(fieldTopic).
				Title("Podcast topic").
				Description("Questions and content are generated for this topic.").
				Placeholder("e.g. quantum computing, black holes...").
				CharLimit(topicCharLimit).
				Validate(validateTopic).
				Value(&m.topic),
		),
	).
		WithTheme(huh.ThemeFunc(huhstyle.Theme)).
		WithAccessible(huhstyle.Accessible())

	// A fresh form does not know the terminal size yet and would fall back to
	// huh's own default width, so the box visibly collapses to that width on every
	// re-arm. Seeding the remembered width keeps the layout stable across a submit.
	m.applyWidth(f)
	return f
}

// applyWidth sizes the form to the page gutter on both sides.
//
// huh lays out to the width it is given, so the gutter has to come out of the
// width rather than being added as indentation afterwards — otherwise the box is
// pushed past the right edge of the terminal.
func (m *Model) applyWidth(f *huh.Form) {
	if m.width == 0 {
		return
	}
	if w := m.width - formGutter; w > minFormWidth {
		f.WithWidth(w)
	}
}

// rearm builds a fresh form and returns its init command.
//
// A huh form that has been submitted sets an internal `quitting` flag: Update
// then returns immediately and View renders an empty string. This app keeps one
// Model per page for the whole session and re-calls Init() every time
// NavigateMsg selects the page, so without re-arming a second visit would paint
// a blank screen with no field and no way to type.
//
// Note there is no StateAborted branch anywhere in this page: the shell consumes
// `esc` and `ctrl+c` before a page ever sees them, so huh's abort/quit bindings
// cannot fire in-app.
func (m *Model) rearm() tea.Cmd {
	m.form = m.newForm()
	// Replay the last known size into the fresh form. A new form has neither width
	// nor height and huh only computes them when it receives a size message, so
	// without this the box renders clipped to huh's defaults: the field row and the
	// bottom border disappear.
	if m.width > 0 {
		m.resizeForm(m.width, m.height)
	}
	return m.form.Init()
}

// resizeForm applies the page width and hands the size to huh.
//
// The width is applied first because huh derives its height from the width it
// wraps at, and it skips its own width calculation once a width has been set —
// which is what keeps the page gutter from being added on top of a full-width box.
func (m *Model) resizeForm(width, height int) tea.Cmd {
	m.applyWidth(m.form)
	form, cmd := m.form.Update(tea.WindowSizeMsg{Width: width, Height: height})
	if f, ok := form.(*huh.Form); ok {
		m.form = f
	}
	return cmd
}

func (m *Model) Init() tea.Cmd {
	return m.rearm()
}

func (m *Model) beginLoad() {
	m.loading = true
	m.err = nil
	m.result = nil
}

func (m *Model) reset() {
	m.topic = ""
	m.result = nil
	m.err = nil
}

func (m *Model) generateTopic() tea.Cmd {
	topic := m.topic
	return func() tea.Msg {
		if m.ctx.API == nil {
			return topicResultMsg{err: fmt.Errorf("API client not configured")}
		}
		result, err := m.ctx.API.GenerateTopic(topic)
		return topicResultMsg{result: result, err: err}
	}
}

func (m *Model) bootstrapDemo() tea.Cmd {
	topic := strings.TrimSpace(m.topic)
	if topic == "" {
		topic = "How airplanes stay in the sky"
	}
	return func() tea.Msg {
		if m.ctx.API == nil {
			return topicResultMsg{err: fmt.Errorf("API client not configured")}
		}
		result, err := m.ctx.API.BootstrapDemo(topic)
		return topicResultMsg{result: result, err: err}
	}
}

func (m *Model) Update(msg tea.Msg) (app.PageModel, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		m.width = msg.Width
		m.height = msg.Height
		return m, m.resizeForm(msg.Width, msg.Height)

	case topicResultMsg:
		m.loading = false
		if msg.err != nil {
			m.err = msg.err
		} else {
			m.result = msg.result
			if !msg.result.Success {
				m.err = fmt.Errorf("%s", msg.result.Message)
			}
		}
		return m, nil

	case spinner.TickMsg:
		var cmd tea.Cmd
		m.spinner, cmd = m.spinner.Update(msg)
		return m, cmd
	}

	if m.loading {
		return m, nil
	}

	// The page's own shortcuts are matched before the form so they cannot be
	// swallowed by the field. Neither ctrl+d nor ctrl+r appears in huh's default
	// keymap (it uses ctrl+e, enter, tab and shift+tab for input), so binding
	// them here shadows nothing.
	if key, ok := msg.(tea.KeyPressMsg); ok {
		switch key.String() {
		case "ctrl+d":
			m.beginLoad()
			return m, tea.Batch(m.spinner.Tick, m.bootstrapDemo())
		case "ctrl+r":
			m.reset()
			return m, m.rearm()
		}
	}

	form, cmd := m.form.Update(msg)
	if f, ok := form.(*huh.Form); ok {
		m.form = f
	}

	if m.form.State == huh.StateCompleted {
		m.beginLoad()
		// Re-arm alongside the request so the field stays on screen while the
		// backend works, and so Enter can start another run afterwards.
		return m, tea.Batch(m.spinner.Tick, m.generateTopic(), m.rearm())
	}

	return m, cmd
}

func (m *Model) View() string {
	th := m.ctx.Theme
	var b strings.Builder

	b.WriteString("  " + th.Header.Render("Topic Explorer") + "\n\n")
	b.WriteString("  " + th.MutedText.Render("Generate questions and content for a podcast topic") + "\n\n")

	// Form. The form renders several lines (border, title, description, field), so
	// the page gutter has to be a padding style rather than a prefix string: a
	// "  " prefix only indents the first line and leaves the box border misaligned
	// with the content it wraps.
	b.WriteString(formBlock.Render(m.form.View()) + "\n\n")

	// Loading state
	if m.loading {
		statusText := "Generating topic content..."
		if m.result == nil && strings.TrimSpace(m.topic) == "" {
			statusText = "Creating demo bundle..."
		}
		b.WriteString(fmt.Sprintf("  %s %s\n",
			m.spinner.View(),
			th.AccentText.Render(statusText),
		))
		b.WriteString("  " + th.MutedText.Render("This may take a minute depending on mode.") + "\n")
		return b.String()
	}

	// Error
	if m.err != nil {
		b.WriteString("  " + th.ErrorText.Render("Error: "+m.err.Error()) + "\n\n")
	}

	// Result
	if m.result != nil && m.result.Success {
		b.WriteString("  " + th.SuccessText.Render("Topic generated successfully!") + "\n\n")

		if m.result.QuestionsFile != "" {
			b.WriteString(fmt.Sprintf("  %s %s\n",
				th.MutedText.Render("Questions file:"),
				th.AccentText.Render(m.result.QuestionsFile),
			))
		}
		if m.result.ContentFile != "" {
			b.WriteString(fmt.Sprintf("  %s %s\n",
				th.MutedText.Render("Content file:"),
				th.AccentText.Render(m.result.ContentFile),
			))
		}
		if m.result.TranscriptFile != "" {
			b.WriteString(fmt.Sprintf("  %s %s\n",
				th.MutedText.Render("Transcript file:"),
				th.AccentText.Render(m.result.TranscriptFile),
			))
		}
		if m.result.AudioFile != "" {
			b.WriteString(fmt.Sprintf("  %s %s\n",
				th.MutedText.Render("Audio file:"),
				th.AccentText.Render(m.result.AudioFile),
			))
		}

		if len(m.result.Questions) > 0 {
			b.WriteString("\n  " + th.Header.Render("Generated Questions") + "\n\n")
			for i, q := range m.result.Questions {
				b.WriteString(fmt.Sprintf("  %s %s\n",
					th.AccentText.Render(fmt.Sprintf("%d.", i+1)),
					lipgloss.NewStyle().Foreground(th.Text).Render(q),
				))
			}
		}
	}

	// Hints
	b.WriteString("\n")
	hints := []string{"[Enter] Generate", "[Ctrl+D] Demo", "[Ctrl+R] Reset", "[Esc] Back to Menu"}
	b.WriteString("  " + th.MutedText.Render(strings.Join(hints, "  |  ")))

	return b.String()
}
