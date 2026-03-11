// internal/theme/theme.go
package theme

import "github.com/charmbracelet/lipgloss"

// Theme holds the aviation/podcast-themed color palette and pre-built styles.
type Theme struct {
	// Colors
	Background lipgloss.Color
	Primary    lipgloss.Color
	Accent     lipgloss.Color
	Alert      lipgloss.Color
	Success    lipgloss.Color
	Text       lipgloss.Color
	TextMuted  lipgloss.Color
	Border     lipgloss.Color

	// Pre-built styles
	TitleBar     lipgloss.Style
	StatusBar    lipgloss.Style
	Panel        lipgloss.Style
	ActiveItem   lipgloss.Style
	InactiveItem lipgloss.Style
	Cursor       lipgloss.Style
	ErrorText    lipgloss.Style
	SuccessText  lipgloss.Style
	AccentText   lipgloss.Style
	MutedText    lipgloss.Style
	Header       lipgloss.Style
}

// New creates a new aviation/podcast themed palette.
func New() *Theme {
	t := &Theme{
		Background: lipgloss.Color("#0B1120"),
		Primary:    lipgloss.Color("#1B3A5C"),
		Accent:     lipgloss.Color("#00D4FF"),
		Alert:      lipgloss.Color("#FF4444"),
		Success:    lipgloss.Color("#00E676"),
		Text:       lipgloss.Color("#E8EDF5"),
		TextMuted:  lipgloss.Color("#7B8CA8"),
		Border:     lipgloss.Color("#2A4A6E"),
	}

	t.TitleBar = lipgloss.NewStyle().
		Background(t.Primary).
		Foreground(t.Text).
		Bold(true).
		Padding(0, 2)

	t.StatusBar = lipgloss.NewStyle().
		Background(t.Primary).
		Foreground(t.TextMuted).
		Padding(0, 1)

	t.Panel = lipgloss.NewStyle().
		Border(lipgloss.RoundedBorder()).
		BorderForeground(t.Border).
		Padding(1, 2)

	t.ActiveItem = lipgloss.NewStyle().
		Foreground(t.Text).
		Background(t.Primary).
		Bold(true).
		Padding(0, 1)

	t.InactiveItem = lipgloss.NewStyle().
		Foreground(t.TextMuted).
		Padding(0, 1)

	t.Cursor = lipgloss.NewStyle().
		Foreground(t.Accent).
		Bold(true)

	t.ErrorText = lipgloss.NewStyle().
		Foreground(t.Alert).
		Bold(true)

	t.SuccessText = lipgloss.NewStyle().
		Foreground(t.Success)

	t.AccentText = lipgloss.NewStyle().
		Foreground(t.Accent).
		Bold(true)

	t.MutedText = lipgloss.NewStyle().
		Foreground(t.TextMuted)

	t.Header = lipgloss.NewStyle().
		Foreground(t.Accent).
		Bold(true).
		Underline(true)

	return t
}

// StatusIndicator returns a colored dot for service status.
func (t *Theme) StatusIndicator(healthy bool) string {
	if healthy {
		return t.SuccessText.Render("●")
	}
	return t.ErrorText.Render("●")
}
