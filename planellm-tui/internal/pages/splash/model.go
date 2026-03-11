package splash

import (
	"math"
	"strings"
	"time"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
	"github.com/jasperan/planellm-tui/internal/app"
	ctx "github.com/jasperan/planellm-tui/internal/context"
	"github.com/jasperan/planellm-tui/internal/theme"
)

const (
	fps         = 20
	totalFrames = 60
	planePhase  = 20
	logoPhase   = 40
)

var airplane = []string{
	`         __|__         `,
	`  --@--@--(_)--@--@--  `,
	`         /   \         `,
}

var logo = []string{
	`        _                  _     _     __  __  `,
	`  _ __ | | __ _ _ __   ___| |   | |   |  \/  | `,
	` | '_ \| |/ _` + "`" + ` | '_ \ / _ \ |   | |   | |\/| | `,
	` | |_) | | (_| | | | |  __/ |___| |___| |  | | `,
	` | .__/|_|\__,_|_| |_|\___|_____|_____|_|  |_| `,
	` |_|                                           `,
}

const subtitle = "Bite-sized podcasts powered by Fish Speech S2"

var waveChars = []rune("▁▂▃▄▅▆▇█▇▆▅▄▃▂▁")

type tickMsg time.Time

// Model is the splash screen page model.
type Model struct {
	ctx    *ctx.Context
	frame  int
	width  int
	height int
}

// New creates a new splash screen model.
func New(c *ctx.Context) *Model {
	return &Model{ctx: c}
}

// Init starts the animation ticker.
func (m *Model) Init() tea.Cmd {
	return tea.Tick(time.Second/fps, func(t time.Time) tea.Msg {
		return tickMsg(t)
	})
}

// Update handles messages for the splash screen.
func (m *Model) Update(msg tea.Msg) (app.PageModel, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		m.width = msg.Width
		m.height = msg.Height
		return m, nil

	case tea.KeyMsg:
		// Any key skips to menu
		return m, func() tea.Msg { return app.NavigateMsg{Page: app.PageMenu} }

	case tickMsg:
		m.frame++
		if m.frame >= totalFrames {
			return m, func() tea.Msg { return app.NavigateMsg{Page: app.PageMenu} }
		}
		return m, tea.Tick(time.Second/fps, func(t time.Time) tea.Msg {
			return tickMsg(t)
		})
	}
	return m, nil
}

// View renders the splash screen.
func (m *Model) View() string {
	if m.width == 0 || m.height == 0 {
		return ""
	}

	th := m.ctx.Theme
	var output strings.Builder

	centerY := m.height / 2

	if m.frame < planePhase {
		// Phase 1: Airplane flies from left to right
		planeX := (m.frame * (m.width + 25)) / planePhase - 25
		planeY := centerY - len(airplane)/2

		for y := 0; y < m.height-1; y++ {
			planeLine := y - planeY
			if planeLine >= 0 && planeLine < len(airplane) {
				// Draw contrail
				trailLen := min(planeX, 50)
				trailStart := max(0, planeX-trailLen)
				line := strings.Repeat(" ", trailStart)
				if trailLen > 0 {
					trail := strings.Repeat("─", min(trailLen, m.width-trailStart))
					line += lipgloss.NewStyle().Foreground(th.TextMuted).Render(trail)
				}
				// Draw airplane at position
				if planeX >= 0 && planeX < m.width+25 {
					planeStr := airplane[planeLine]
					visibleStart := max(0, -planeX)
					visibleEnd := min(len(planeStr), m.width-max(0, planeX))
					if visibleEnd > visibleStart {
						line = padToWidth(line, max(0, planeX))
						line += lipgloss.NewStyle().Foreground(th.Accent).Bold(true).Render(planeStr[visibleStart:visibleEnd])
					}
				}
				output.WriteString(truncateToWidth(line, m.width))
			}
			output.WriteString("\n")
		}
	} else if m.frame < logoPhase {
		// Phase 2: Logo assembles letter by letter
		progress := float64(m.frame-planePhase) / float64(logoPhase-planePhase)
		logoY := centerY - len(logo)/2

		for y := 0; y < m.height-1; y++ {
			logoLine := y - logoY
			if logoLine >= 0 && logoLine < len(logo) {
				line := logo[logoLine]
				visibleChars := int(float64(len(line)) * progress)
				if visibleChars > len(line) {
					visibleChars = len(line)
				}
				visible := line[:visibleChars]

				var styled string
				if progress > 0.7 {
					styled = lipgloss.NewStyle().Foreground(th.Accent).Bold(true).Render(visible)
				} else {
					styled = lipgloss.NewStyle().Foreground(th.Primary).Render(visible)
				}
				padding := (m.width - len(line)) / 2
				if padding < 0 {
					padding = 0
				}
				output.WriteString(strings.Repeat(" ", padding) + styled)
			} else if y == logoY+len(logo)+1 && progress > 0.8 {
				sub := lipgloss.NewStyle().Foreground(th.TextMuted).Render(subtitle)
				padding := (m.width - len(subtitle)) / 2
				if padding < 0 {
					padding = 0
				}
				output.WriteString(strings.Repeat(" ", padding) + sub)
			}
			output.WriteString("\n")
		}
	} else {
		// Phase 3: Hold complete logo with subtitle and waveform
		logoY := centerY - len(logo)/2 - 1
		accentStyle := lipgloss.NewStyle().Foreground(th.Accent).Bold(true)

		for y := 0; y < m.height-1; y++ {
			logoLine := y - logoY
			if logoLine >= 0 && logoLine < len(logo) {
				styled := accentStyle.Render(logo[logoLine])
				padding := (m.width - len(logo[logoLine])) / 2
				if padding < 0 {
					padding = 0
				}
				output.WriteString(strings.Repeat(" ", padding) + styled)
			} else if y == logoY+len(logo)+1 {
				sub := lipgloss.NewStyle().Foreground(th.TextMuted).Render(subtitle)
				padding := (m.width - len(subtitle)) / 2
				if padding < 0 {
					padding = 0
				}
				output.WriteString(strings.Repeat(" ", padding) + sub)
			} else if y == logoY+len(logo)+3 {
				// Audio waveform animation
				wave := renderWaveform(m.frame, m.width, th)
				output.WriteString(wave)
			}
			output.WriteString("\n")
		}
	}

	return output.String()
}

// renderWaveform creates a scrolling audio waveform animation.
func renderWaveform(frame, width int, th *theme.Theme) string {
	if width <= 0 {
		return ""
	}
	waveLen := len(waveChars)
	var b strings.Builder
	padding := (width - min(width, 60)) / 2
	b.WriteString(strings.Repeat(" ", padding))

	displayWidth := min(width-padding*2, 60)
	for i := 0; i < displayWidth; i++ {
		// Scrolling wave with sine modulation for organic feel
		idx := (i + frame*2) % waveLen
		amplitude := 0.5 + 0.5*math.Sin(float64(i+frame)/4.0)
		charIdx := int(float64(idx) * amplitude)
		if charIdx >= waveLen {
			charIdx = waveLen - 1
		}
		if charIdx < 0 {
			charIdx = 0
		}
		b.WriteRune(waveChars[charIdx])
	}

	styled := lipgloss.NewStyle().Foreground(th.Accent).Render(b.String())
	return styled
}

func padToWidth(s string, width int) string {
	current := len(s)
	if current < width {
		return s + strings.Repeat(" ", width-current)
	}
	return s
}

func truncateToWidth(s string, width int) string {
	if len(s) > width {
		return s[:width]
	}
	return s
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
