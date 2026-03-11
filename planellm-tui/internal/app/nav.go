package app

// Page identifies a TUI page.
type Page int

const (
	PageSplash Page = iota
	PageMenu
	PageTopic
	PageTranscript
	PageAudio
	PageStatus
)

func (p Page) String() string {
	names := []string{
		"Splash", "Menu", "Topic Explorer", "Transcript Writer",
		"Audio Generator", "System Status",
	}
	if int(p) < len(names) {
		return names[p]
	}
	return "Unknown"
}
