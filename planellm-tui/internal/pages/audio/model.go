package audio

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

type audioResultMsg struct {
	result *ctx.AudioResult
	err    error
}

var ttsModels = []string{"fish", "parler", "bark"}

var ttsDescriptions = map[string]string{
	"fish":   "Fish Speech S2 - High quality neural TTS with emotion control and voice cloning",
	"parler": "Parler TTS - Fast and lightweight text-to-speech with natural prosody",
	"bark":   "Bark - Highly realistic multilingual speech with sound effects support",
}

type Model struct {
	ctx        *ctx.Context
	spinner    spinner.Model
	files      []string
	cursor     int
	ttsIdx     int
	result     *ctx.AudioResult
	loading    bool
	err        error
	loaded     bool
	width      int
	height     int
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

func (m *Model) generateAudio() tea.Cmd {
	file := m.files[m.cursor]
	model := ttsModels[m.ttsIdx]
	return func() tea.Msg {
		result, err := m.ctx.API.GenerateAudio(file, model, "", "")
		return audioResultMsg{result: result, err: err}
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
			m.files = msg.files.Transcripts
		}
		return m, nil

	case audioResultMsg:
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
		case "tab":
			m.ttsIdx = (m.ttsIdx + 1) % len(ttsModels)
		case "enter":
			if len(m.files) > 0 && m.cursor < len(m.files) {
				m.loading = true
				m.err = nil
				m.result = nil
				return m, tea.Batch(m.spinner.Tick, m.generateAudio())
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

	b.WriteString("  " + th.Header.Render("Audio Generator") + "\n\n")

	if !m.loaded {
		b.WriteString("  " + th.MutedText.Render("Loading files...") + "\n")
		return b.String()
	}

	if m.err != nil && m.result == nil && !m.loading {
		b.WriteString("  " + th.ErrorText.Render("Error: "+m.err.Error()) + "\n\n")
	}

	// TTS model selector
	b.WriteString("  " + th.MutedText.Render("TTS Model [Tab to cycle]:") + "\n")
	b.WriteString("  ")
	for i, model := range ttsModels {
		if i == m.ttsIdx {
			b.WriteString(th.ActiveItem.Render(" " + model + " "))
		} else {
			b.WriteString(th.InactiveItem.Render(" " + model + " "))
		}
		if i < len(ttsModels)-1 {
			b.WriteString("  ")
		}
	}
	b.WriteString("\n")

	// Model description panel
	currentModel := ttsModels[m.ttsIdx]
	desc := ttsDescriptions[currentModel]
	b.WriteString("  " + th.MutedText.Render(desc) + "\n\n")

	// File list
	if len(m.files) == 0 {
		b.WriteString("  " + th.MutedText.Render("No transcript files found. Create a transcript first.") + "\n")
	} else {
		b.WriteString("  " + th.MutedText.Render("Select a transcript file:") + "\n\n")
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
			th.AccentText.Render("Generating audio with "+ttsModels[m.ttsIdx]+"..."),
		))
		b.WriteString("  " + th.MutedText.Render("This may take several minutes") + "\n")
		return b.String()
	}

	// Result
	if m.result != nil && m.result.Success {
		b.WriteString("\n  " + th.SuccessText.Render("Audio generated successfully!") + "\n\n")

		if m.result.AudioFile != "" {
			b.WriteString(fmt.Sprintf("  %s %s\n",
				th.MutedText.Render("Output file:"),
				th.AccentText.Render(m.result.AudioFile),
			))
		}
		if m.result.Message != "" {
			b.WriteString(fmt.Sprintf("  %s %s\n",
				th.MutedText.Render("Message:"),
				lipgloss.NewStyle().Foreground(th.Text).Render(m.result.Message),
			))
		}
	}

	// Hints
	b.WriteString("\n")
	hints := []string{"[Enter] Generate", "[Tab] Cycle TTS", "[r] Refresh Files", "[Esc] Back"}
	b.WriteString("  " + th.MutedText.Render(strings.Join(hints, "  |  ")))

	return b.String()
}
