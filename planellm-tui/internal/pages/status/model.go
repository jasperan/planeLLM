package status

import (
	"fmt"
	"strings"
	"time"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
	"github.com/jasperan/planellm-tui/internal/app"
	ctx "github.com/jasperan/planellm-tui/internal/context"
)

type statusResultMsg struct {
	status *ctx.SystemStatus
	err    error
}

type refreshMsg struct{}

type Model struct {
	ctx    *ctx.Context
	status *ctx.SystemStatus
	err    error
	width  int
	height int
}

func New(c *ctx.Context) *Model {
	return &Model{ctx: c}
}

func (m *Model) Init() tea.Cmd {
	return tea.Batch(m.fetchStatus(), m.tickRefresh())
}

func (m *Model) fetchStatus() tea.Cmd {
	return func() tea.Msg {
		if m.ctx.API == nil {
			return statusResultMsg{err: fmt.Errorf("API client not configured")}
		}
		s, err := m.ctx.API.GetStatus()
		return statusResultMsg{status: s, err: err}
	}
}

func (m *Model) tickRefresh() tea.Cmd {
	return tea.Tick(3*time.Second, func(t time.Time) tea.Msg {
		return refreshMsg{}
	})
}

func (m *Model) Update(msg tea.Msg) (app.PageModel, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		m.width = msg.Width
		m.height = msg.Height
		return m, nil
	case tea.KeyMsg:
		if msg.String() == "r" {
			return m, m.fetchStatus()
		}
	case refreshMsg:
		return m, tea.Batch(m.fetchStatus(), m.tickRefresh())
	case statusResultMsg:
		m.status = msg.status
		m.err = msg.err
	}
	return m, nil
}

func (m *Model) View() string {
	th := m.ctx.Theme

	if m.err != nil {
		return th.Panel.Render(
			th.ErrorText.Render("Error: "+m.err.Error()) + "\n\n" +
				th.MutedText.Render("[r] Refresh  |  Is planeLLM API running on :7880?"),
		)
	}

	if m.status == nil {
		return th.Panel.Render(th.MutedText.Render("Loading..."))
	}

	s := m.status

	servicePanel := func(name string, healthy bool) string {
		indicator := th.StatusIndicator(healthy)
		label := lipgloss.NewStyle().Foreground(th.Text).Render(name)
		return fmt.Sprintf("  %s %s", indicator, label)
	}

	services := []string{
		th.Header.Render("Services"),
		"",
		servicePanel("OCI GenAI SDK", s.OCIConfig),
		servicePanel("FFmpeg", s.FFmpeg),
		servicePanel("Fish Audio SDK", s.FishSDK),
	}

	panelW := 35
	if m.width > 0 {
		panelW = (m.width - 6) / 2
		if panelW < 30 {
			panelW = 30
		}
	}

	servicesBlock := th.Panel.Width(panelW).Render(strings.Join(services, "\n"))

	resourceLines := []string{
		th.Header.Render("Resources"),
		"",
		fmt.Sprintf("  %s %s",
			th.MutedText.Render(fmt.Sprintf("%-16s", "Total files")),
			th.AccentText.Render(fmt.Sprintf("%d", s.ResourcesCount)),
		),
	}
	resourceBlock := th.Panel.Width(panelW).Render(strings.Join(resourceLines, "\n"))

	grid := lipgloss.JoinHorizontal(lipgloss.Top, servicesBlock, "  ", resourceBlock)
	hint := th.MutedText.Render("\n  [r] Refresh  |  Auto-refresh: 3s")

	return grid + hint
}
