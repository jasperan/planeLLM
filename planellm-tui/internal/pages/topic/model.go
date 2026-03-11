package topic

import (
	"fmt"
	"strings"

	"github.com/charmbracelet/bubbles/spinner"
	"github.com/charmbracelet/bubbles/textinput"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
	"github.com/jasperan/planellm-tui/internal/app"
	ctx "github.com/jasperan/planellm-tui/internal/context"
)

type topicResultMsg struct {
	result *ctx.TopicResult
	err    error
}

type Model struct {
	ctx       *ctx.Context
	input     textinput.Model
	spinner   spinner.Model
	result    *ctx.TopicResult
	loading   bool
	err       error
	width     int
	height    int
}

func New(c *ctx.Context) *Model {
	ti := textinput.New()
	ti.Placeholder = "Enter a podcast topic (e.g. quantum computing, black holes...)"
	ti.CharLimit = 200
	ti.Width = 60
	ti.Focus()

	s := spinner.New()
	s.Spinner = spinner.Dot
	s.Style = lipgloss.NewStyle().Foreground(lipgloss.Color("#00D4FF"))

	return &Model{
		ctx:     c,
		input:   ti,
		spinner: s,
	}
}

func (m *Model) Init() tea.Cmd {
	return textinput.Blink
}

func (m *Model) generateTopic() tea.Cmd {
	topic := m.input.Value()
	return func() tea.Msg {
		if m.ctx.API == nil {
			return topicResultMsg{err: fmt.Errorf("API client not configured")}
		}
		result, err := m.ctx.API.GenerateTopic(topic)
		return topicResultMsg{result: result, err: err}
	}
}

func (m *Model) Update(msg tea.Msg) (app.PageModel, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		m.width = msg.Width
		m.height = msg.Height
		return m, nil

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

	case tea.KeyMsg:
		if m.loading {
			return m, nil
		}
		switch msg.String() {
		case "enter":
			if m.input.Value() != "" && !m.loading {
				m.loading = true
				m.err = nil
				m.result = nil
				return m, tea.Batch(m.spinner.Tick, m.generateTopic())
			}
		case "ctrl+r":
			// Reset
			m.input.SetValue("")
			m.result = nil
			m.err = nil
			m.input.Focus()
			return m, textinput.Blink
		}
	}

	if !m.loading {
		var cmd tea.Cmd
		m.input, cmd = m.input.Update(msg)
		return m, cmd
	}

	return m, nil
}

func (m *Model) View() string {
	th := m.ctx.Theme
	var b strings.Builder

	b.WriteString("  " + th.Header.Render("Topic Explorer") + "\n\n")
	b.WriteString("  " + th.MutedText.Render("Generate questions and content for a podcast topic") + "\n\n")

	// Input
	b.WriteString("  " + m.input.View() + "\n\n")

	// Loading state
	if m.loading {
		b.WriteString(fmt.Sprintf("  %s %s\n",
			m.spinner.View(),
			th.AccentText.Render("Generating topic content..."),
		))
		b.WriteString("  " + th.MutedText.Render("This may take a minute (LLM generation)") + "\n")
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
	hints := []string{"[Enter] Generate", "[Ctrl+R] Reset", "[Esc] Back to Menu"}
	b.WriteString("  " + th.MutedText.Render(strings.Join(hints, "  |  ")))

	return b.String()
}
