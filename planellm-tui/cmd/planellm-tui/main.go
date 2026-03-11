package main

import (
	"fmt"
	"os"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/jasperan/planellm-tui/internal/app"
	ctx "github.com/jasperan/planellm-tui/internal/context"
	"github.com/jasperan/planellm-tui/internal/pages/audio"
	"github.com/jasperan/planellm-tui/internal/pages/menu"
	"github.com/jasperan/planellm-tui/internal/pages/splash"
	"github.com/jasperan/planellm-tui/internal/pages/status"
	"github.com/jasperan/planellm-tui/internal/pages/topic"
	"github.com/jasperan/planellm-tui/internal/pages/transcript"
	"github.com/jasperan/planellm-tui/internal/services"
	"github.com/jasperan/planellm-tui/internal/theme"
)

func envOr(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

func main() {
	th := theme.New()
	apiClient := services.NewAPIClient(envOr("PLANELLM_API_URL", "http://localhost:7880"))

	c := &ctx.Context{
		Theme: th,
		API:   apiClient,
	}

	pages := map[app.Page]app.PageModel{
		app.PageSplash:     splash.New(c),
		app.PageMenu:       menu.New(c),
		app.PageTopic:      topic.New(c),
		app.PageTranscript: transcript.New(c),
		app.PageAudio:      audio.New(c),
		app.PageStatus:     status.New(c),
	}

	m := app.New(c, pages)
	p := tea.NewProgram(m, tea.WithAltScreen())
	if _, err := p.Run(); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
}
