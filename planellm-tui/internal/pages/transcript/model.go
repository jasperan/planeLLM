package transcript

import (
	"fmt"
	"strings"

	"github.com/charmbracelet/bubbles/spinner"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
	"github.com/jasperan/planellm-tui/internal/app"
	ctx "github.com/jasperan/planellm-tui/internal/context"
)

type filesLoadedMsg struct {
	files *ctx.FileList
	err   error
}

type transcriptResultMsg struct {
	result *ctx.TranscriptResult
	err    error
}

type Model struct {
	ctx      *ctx.Context
	spinner  spinner.Model
	files    []string
	cursor   int
	detailed bool
	result   *ctx.TranscriptResult
	loading  bool
	err      error
	loaded   bool
	width    int
	height   int
}

func New(c *ctx.Context) *Model {
	s := spinner.New()
	s.Spinner = spinner.Dot
	s.Style = lipgloss.NewStyle().Foreground(lipgloss.Color("#00D4FF"))

	return &Model{
		ctx:     c,
		spinner: s,
	}
}

func (m *Model) Init() tea.Cmd {
	return tea.Batch(m.spinner.Tick, m.loadFiles())
}

func (m *Model) loadFiles() tea.Cmd {
	return func() tea.Msg {
		if m.ctx.API == nil {
			return filesLoadedMsg{err: fmt.Errorf("API client not configured")}
		}
		files, err := m.ctx.API.ListFiles()
		return filesLoadedMsg{files: files, err: err}
	}
}

func (m *Model) createTranscript() tea.Cmd {
	file := m.files[m.cursor]
	detailed := m.detailed
	return func() tea.Msg {
		result, err := m.ctx.API.CreateTranscript(file, detailed)
		return transcriptResultMsg{result: result, err: err}
	}
}

func (m *Model) Update(msg tea.Msg) (app.PageModel, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		m.width = msg.Width
		m.height = msg.Height
		return m, nil

	case filesLoadedMsg:
		m.loaded = true
		if msg.err != nil {
			m.err = msg.err
		} else if msg.files != nil {
			m.files = msg.files.Content
		}
		return m, nil

	case transcriptResultMsg:
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
		case "up", "k":
			if m.cursor > 0 {
				m.cursor--
			}
		case "down", "j":
			if m.cursor < len(m.files)-1 {
				m.cursor++
			}
		case "d":
			m.detailed = !m.detailed
		case "enter":
			if len(m.files) > 0 && m.cursor < len(m.files) {
				m.loading = true
				m.err = nil
				m.result = nil
				return m, tea.Batch(m.spinner.Tick, m.createTranscript())
			}
		case "r":
			m.result = nil
			m.err = nil
			return m, m.loadFiles()
		}
	}
	return m, nil
}

func (m *Model) View() string {
	th := m.ctx.Theme
	var b strings.Builder

	b.WriteString("  " + th.Header.Render("Transcript Writer") + "\n\n")

	if !m.loaded {
		b.WriteString("  " + th.MutedText.Render("Loading files...") + "\n")
		return b.String()
	}

	if m.err != nil && m.result == nil && !m.loading {
		b.WriteString("  " + th.ErrorText.Render("Error: "+m.err.Error()) + "\n\n")
	}

	// Detailed mode toggle
	detailedIcon := "[ ]"
	if m.detailed {
		detailedIcon = "[x]"
	}
	b.WriteString(fmt.Sprintf("  %s %s %s\n\n",
		th.MutedText.Render("Detailed mode:"),
		th.AccentText.Render(detailedIcon),
		th.MutedText.Render("[d] to toggle"),
	))

	// File list
	if len(m.files) == 0 {
		b.WriteString("  " + th.MutedText.Render("No content files found. Generate a topic first.") + "\n")
	} else {
		b.WriteString("  " + th.MutedText.Render("Select a content file:") + "\n\n")
		for i, file := range m.files {
			cursor := "  "
			style := th.InactiveItem
			if i == m.cursor {
				cursor = th.Cursor.Render("▸ ")
				style = th.ActiveItem
			}
			b.WriteString(cursor + style.Render(file) + "\n")
		}
	}

	// Loading state
	if m.loading {
		b.WriteString(fmt.Sprintf("\n  %s %s\n",
			m.spinner.View(),
			th.AccentText.Render("Creating transcript..."),
		))
		b.WriteString("  " + th.MutedText.Render("This may take a minute (LLM generation)") + "\n")
		return b.String()
	}

	// Result
	if m.result != nil && m.result.Success {
		b.WriteString("\n  " + th.SuccessText.Render("Transcript created!") + "\n\n")

		if m.result.TranscriptFile != "" {
			b.WriteString(fmt.Sprintf("  %s %s\n",
				th.MutedText.Render("Output file:"),
				th.AccentText.Render(m.result.TranscriptFile),
			))
		}

		if m.result.TranscriptPreview != "" {
			b.WriteString("\n  " + th.Header.Render("Preview") + "\n\n")
			// Show preview lines
			lines := strings.Split(m.result.TranscriptPreview, "\n")
			maxLines := 15
			if len(lines) > maxLines {
				lines = lines[:maxLines]
			}
			for _, line := range lines {
				b.WriteString("  " + lipgloss.NewStyle().Foreground(th.Text).Render(line) + "\n")
			}
			if len(strings.Split(m.result.TranscriptPreview, "\n")) > maxLines {
				b.WriteString("  " + th.MutedText.Render("...") + "\n")
			}
		}
	}

	// Hints
	b.WriteString("\n")
	hints := []string{"[Enter] Create", "[d] Toggle Detailed", "[r] Refresh Files", "[Esc] Back"}
	b.WriteString("  " + th.MutedText.Render(strings.Join(hints, "  |  ")))

	return b.String()
}
