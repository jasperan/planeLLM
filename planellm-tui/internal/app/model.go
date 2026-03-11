package app

import (
	"github.com/charmbracelet/lipgloss"
	ctx "github.com/jasperan/planellm-tui/internal/context"
	"github.com/jasperan/planellm-tui/internal/theme"

	tea "github.com/charmbracelet/bubbletea"
)

// PageModel is the interface all pages implement.
type PageModel interface {
	Init() tea.Cmd
	Update(tea.Msg) (PageModel, tea.Cmd)
	View() string
}

// NavigateMsg tells the app to switch pages.
type NavigateMsg struct {
	Page Page
}

// Model is the root app model.
type Model struct {
	ctx     *ctx.Context
	current Page
	pages   map[Page]PageModel
	width   int
	height  int
}

// New creates the root model with all pages registered.
func New(c *ctx.Context, pages map[Page]PageModel) Model {
	return Model{
		ctx:     c,
		current: PageSplash,
		pages:   pages,
	}
}

func (m Model) Init() tea.Cmd {
	if p, ok := m.pages[m.current]; ok {
		return p.Init()
	}
	return nil
}

func (m Model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		m.width = msg.Width
		m.height = msg.Height
		m.ctx.Width = msg.Width
		m.ctx.Height = msg.Height
		// Forward to current page
		if p, ok := m.pages[m.current]; ok {
			updated, cmd := p.Update(msg)
			m.pages[m.current] = updated
			return m, cmd
		}
		return m, nil

	case NavigateMsg:
		m.current = msg.Page
		if p, ok := m.pages[m.current]; ok {
			return m, p.Init()
		}
		return m, nil

	case tea.KeyMsg:
		// Global quit
		if msg.String() == "ctrl+c" {
			return m, tea.Quit
		}
		// Escape goes back to menu from any page (except splash/menu)
		if msg.String() == "esc" && m.current != PageSplash && m.current != PageMenu {
			m.current = PageMenu
			return m, m.pages[PageMenu].Init()
		}
	}

	// Forward to current page
	if p, ok := m.pages[m.current]; ok {
		updated, cmd := p.Update(msg)
		m.pages[m.current] = updated
		return m, cmd
	}
	return m, nil
}

func (m Model) View() string {
	if m.width == 0 {
		return "Loading..."
	}

	page, ok := m.pages[m.current]
	if !ok {
		return "Page not found"
	}

	// Splash has no chrome
	if m.current == PageSplash {
		return page.View()
	}

	th := m.ctx.Theme

	// Title bar
	titleBar := th.TitleBar.Width(m.width).Render(
		" planeLLM  |  " + m.current.String(),
	)

	// Status bar
	navHints := " <- Esc Menu  |  Up/Down/jk Navigate  |  Enter Select  |  q Quit"
	statusBar := th.StatusBar.Width(m.width).Render(navHints)

	// Page content — fill remaining height
	contentHeight := m.height - lipgloss.Height(titleBar) - lipgloss.Height(statusBar)
	content := lipgloss.NewStyle().
		Width(m.width).
		Height(contentHeight).
		Render(page.View())

	return titleBar + "\n" + content + "\n" + statusBar
}

// NewTheme is a convenience re-export.
func NewTheme() *theme.Theme {
	return theme.New()
}
