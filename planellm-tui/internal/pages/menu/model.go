package menu

import (
	"strings"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
	"github.com/jasperan/planellm-tui/internal/app"
	ctx "github.com/jasperan/planellm-tui/internal/context"
)

var menuLogo = []string{
	`        _                  _     _     __  __  `,
	`  _ __ | | __ _ _ __   ___| |   | |   |  \/  | `,
	` | '_ \| |/ _` + "`" + ` | '_ \ / _ \ |   | |   | |\/| | `,
	` | |_) | | (_| | | | |  __/ |___| |___| |  | | `,
	` | .__/|_|\__,_|_| |_|\___|_____|_____|_|  |_| `,
	` |_|                                           `,
}

type menuItem struct {
	label string
	page  app.Page
}

var items = []menuItem{
	{"Topic Explorer", app.PageTopic},
	{"Transcript Writer", app.PageTranscript},
	{"Audio Generator", app.PageAudio},
	{"System Status", app.PageStatus},
}

type Model struct {
	ctx    *ctx.Context
	cursor int
	width  int
	height int
}

func New(c *ctx.Context) *Model {
	return &Model{ctx: c}
}

func (m *Model) Init() tea.Cmd { return nil }

func (m *Model) Update(msg tea.Msg) (app.PageModel, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		m.width = msg.Width
		m.height = msg.Height
		return m, nil

	case tea.KeyMsg:
		switch msg.String() {
		case "q", "esc":
			return m, tea.Quit
		case "up", "k":
			if m.cursor > 0 {
				m.cursor--
			}
		case "down", "j":
			if m.cursor < len(items)-1 {
				m.cursor++
			}
		case "enter", " ":
			target := items[m.cursor].page
			return m, func() tea.Msg { return app.NavigateMsg{Page: target} }
		}
	}
	return m, nil
}

func (m *Model) View() string {
	th := m.ctx.Theme

	// Logo in accent color
	logoStyle := lipgloss.NewStyle().Foreground(th.Accent).Bold(true)
	var logoLines []string
	for _, line := range menuLogo {
		logoLines = append(logoLines, logoStyle.Render(line))
	}
	logoBlock := strings.Join(logoLines, "\n")

	// Menu items with cursor
	var menuLines []string
	for i, item := range items {
		cursor := "  "
		style := th.InactiveItem
		if i == m.cursor {
			cursor = th.Cursor.Render("▸ ")
			style = th.ActiveItem
		}
		menuLines = append(menuLines, cursor+style.Render(item.label))
	}
	menuBlock := strings.Join(menuLines, "\n")

	// Quit hint
	quitHint := th.MutedText.Render("\n  [q] Quit")

	// Compose
	content := logoBlock + "\n\n" + menuBlock + quitHint

	// Center vertically and horizontally
	contentHeight := lipgloss.Height(content)
	topPad := 0
	if m.height > 0 {
		topPad = (m.height - contentHeight) / 2
		if topPad < 0 {
			topPad = 0
		}
	}

	width := m.width
	if width == 0 {
		width = 80
	}

	centered := lipgloss.NewStyle().
		Width(width).
		Align(lipgloss.Center).
		PaddingTop(topPad).
		Render(content)

	return centered
}
