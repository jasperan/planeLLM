package app

import (
	"strings"
	"testing"

	tea "charm.land/bubbletea/v2"
	ctx "github.com/jasperan/planellm-tui/internal/context"
	"github.com/jasperan/planellm-tui/internal/theme"
)

// This module had no tests at all before the v2 migration. These cover the shell
// behaviour most likely to regress silently: the alternate screen is now a
// property of the returned View rather than a global program option, so it has to
// be set on every return path, and key routing runs through tea.KeyPressMsg.

// fakePage is a stand-in page. Its View is a single marker line so tests can tell
// which page rendered.
type fakePage struct {
	id      string
	lastMsg tea.Msg
	inits   int
}

// namedKey builds the message the runtime delivers for a named key and asserts it
// really reports that name, so a test cannot silently exercise the wrong branch.
func namedKey(t *testing.T, name string, k tea.KeyPressMsg) tea.KeyPressMsg {
	t.Helper()
	if got := k.String(); got != name {
		t.Fatalf("key construction: message reports %q but the test asked for %q", got, name)
	}
	return k
}

// initMsg is returned by fakePage.Init so tests can tell whether the shell really
// propagated the page's Init command, rather than merely calling Init.
type initMsg struct{ id string }

func (f *fakePage) Init() tea.Cmd {
	f.inits++
	id := f.id
	return func() tea.Msg { return initMsg{id: id} }
}

func (f *fakePage) Update(msg tea.Msg) (PageModel, tea.Cmd) {
	f.lastMsg = msg
	return f, nil
}

func (f *fakePage) View() string { return "PAGE:" + f.id }

func newShell(t *testing.T, w, h int) (Model, map[Page]*fakePage) {
	t.Helper()
	fakes := map[Page]*fakePage{}
	pages := map[Page]PageModel{}
	for _, p := range []Page{PageSplash, PageMenu, PageTopic, PageTranscript, PageAudio, PageStatus} {
		f := &fakePage{id: p.String()}
		fakes[p] = f
		pages[p] = f
	}
	return Model{
		ctx:     &ctx.Context{Theme: theme.New()},
		current: PageSplash,
		pages:   pages,
		width:   w,
		height:  h,
	}, fakes
}

// The alternate screen is set inside View() in v2, and AltScreen is re-evaluated
// on every render. A single return path that omitted it would drop the terminal
// out of the alternate screen and re-enter it on the next frame. v1 expressed
// this as a global program option, which could not be missed this way — so this
// test is the thing that replaces that guarantee.
func TestViewRequestsAltScreenOnEveryPath(t *testing.T) {
	t.Run("zero size", func(t *testing.T) {
		m, _ := newShell(t, 0, 0)
		if v := m.View(); !v.AltScreen {
			t.Errorf("AltScreen = false on the zero-size path (content %q)", v.Content)
		}
	})

	t.Run("missing page", func(t *testing.T) {
		m, _ := newShell(t, 80, 24)
		m.current = Page(9999)
		if v := m.View(); !v.AltScreen {
			t.Errorf("AltScreen = false on the missing-page path (content %q)", v.Content)
		}
	})

	t.Run("splash", func(t *testing.T) {
		m, _ := newShell(t, 80, 24)
		m.current = PageSplash
		if v := m.View(); !v.AltScreen {
			t.Errorf("AltScreen = false on the splash path (content %q)", v.Content)
		}
	})

	t.Run("chrome page", func(t *testing.T) {
		m, _ := newShell(t, 80, 24)
		m.current = PageTopic
		if v := m.View(); !v.AltScreen {
			t.Errorf("AltScreen = false on the chrome path (content %q)", v.Content)
		}
	})

	// Every page, so no page can be the one path that forgot.
	for _, p := range []Page{PageSplash, PageMenu, PageTopic, PageTranscript, PageAudio, PageStatus} {
		m, _ := newShell(t, 80, 24)
		m.current = p
		if v := m.View(); !v.AltScreen {
			t.Errorf("page %v: AltScreen = false", p)
		}
	}
}

func TestViewRendersTheCurrentPage(t *testing.T) {
	m, _ := newShell(t, 80, 24)
	m.current = PageTopic
	if got := m.View().Content; !strings.Contains(got, "PAGE:"+PageTopic.String()) {
		t.Errorf("View() does not render the current page; got:\n%s", got)
	}
}

// The splash deliberately has no chrome; every other page must have the title bar
// and the status bar, or navigation guidance disappears.
func TestSplashHasNoChromeAndOtherPagesDo(t *testing.T) {
	splash, _ := newShell(t, 80, 24)
	splash.current = PageSplash
	if got := splash.View().Content; strings.Contains(got, "planellm-tui") || strings.Contains(got, "Esc Menu") {
		t.Errorf("splash should have no chrome; got:\n%s", got)
	}

	page, _ := newShell(t, 80, 24)
	page.current = PageMenu
	got := page.View().Content
	if !strings.Contains(got, "planeLLM") {
		t.Errorf("chrome page is missing the title bar; got:\n%s", got)
	}
	if !strings.Contains(got, "Esc Menu") {
		t.Errorf("chrome page is missing the status bar; got:\n%s", got)
	}
}

// A key RELEASE must not trigger the shell's shortcuts either. In v2 tea.KeyMsg is an
// interface covering both press and release, and a release reports the same name as its
// press, so a bare `case tea.KeyMsg:` would let a release quit or navigate.
func TestKeyReleaseDoesNotTriggerShellShortcuts(t *testing.T) {
	release := tea.KeyReleaseMsg{Code: 'c', Mod: tea.ModCtrl}

	for _, p := range []Page{PageSplash, PageMenu, PageTopic, PageTranscript, PageAudio, PageStatus} {
		m, fakes := newShell(t, 80, 24)
		m.current = p

		next, cmd := m.Update(release)

		if cmd != nil {
			if _, quit := cmd().(tea.QuitMsg); quit {
				t.Errorf("page %v: releasing ctrl+c quit the program; a release must not act", p)
			}
		}
		if got := next.(Model).current; got != p {
			t.Errorf("page %v: releasing ctrl+c changed the page to %v", p, got)
		}

		// It must also have been forwarded to the page, like any other unhandled message.
		if _, ok := fakes[p].lastMsg.(tea.KeyReleaseMsg); !ok {
			t.Errorf("page %v: releasing ctrl+c was not forwarded (last msg %T)", p, fakes[p].lastMsg)
		}
	}
}

func TestCtrlCQuitsFromEveryPage(t *testing.T) {
	// v2 has no tea.KeyCtrlC constant: a control chord is Code + ModCtrl, and
	// Key.Text is empty for it. The namedKey self-check asserts this really
	// reports "ctrl+c", since a raw Code:3 reports "\x03" instead and would not
	// match the shell's check.
	ctrlC := namedKey(t, "ctrl+c", tea.KeyPressMsg{Code: 'c', Mod: tea.ModCtrl})

	for _, p := range []Page{PageSplash, PageMenu, PageTopic, PageTranscript, PageAudio, PageStatus} {
		m, _ := newShell(t, 80, 24)
		m.current = p
		_, cmd := m.Update(ctrlC)
		if cmd == nil {
			t.Errorf("page %v: ctrl+c produced no command, want tea.Quit", p)
			continue
		}
		if _, ok := cmd().(tea.QuitMsg); !ok {
			t.Errorf("page %v: ctrl+c did not produce tea.QuitMsg", p)
		}
	}
}

// Escape is the documented way back to the menu from a page.
func TestEscapeReturnsToTheMenuFromAPage(t *testing.T) {
	for _, p := range []Page{PageTopic, PageTranscript, PageAudio, PageStatus} {
		m, _ := newShell(t, 80, 24)
		m.current = p
		next, _ := m.Update(tea.KeyPressMsg{Code: tea.KeyEsc})
		if got := next.(Model).current; got != PageMenu {
			t.Errorf("esc from %v landed on %v, want %v", p, got, PageMenu)
		}
	}
}

// ...but not from the splash or the menu, where escape is the page's own business.
func TestEscapeIsForwardedFromSplashAndMenu(t *testing.T) {
	for _, p := range []Page{PageSplash, PageMenu} {
		m, fakes := newShell(t, 80, 24)
		m.current = p
		next, _ := m.Update(tea.KeyPressMsg{Code: tea.KeyEsc})
		if got := next.(Model).current; got != p {
			t.Errorf("esc from %v changed the page to %v", p, got)
		}
		if _, ok := fakes[p].lastMsg.(tea.KeyPressMsg); !ok {
			t.Errorf("esc from %v was not forwarded to the page (last msg %T)", p, fakes[p].lastMsg)
		}
	}
}

func TestWindowSizeIsForwardedToTheCurrentPage(t *testing.T) {
	m, fakes := newShell(t, 80, 24)
	m.current = PageTopic

	next, _ := m.Update(tea.WindowSizeMsg{Width: 120, Height: 40})
	got := next.(Model)

	if got.width != 120 || got.height != 40 {
		t.Errorf("shell size = %dx%d, want 120x40", got.width, got.height)
	}
	// The size is also published on the shared context, which pages read.
	if got.ctx.Width != 120 || got.ctx.Height != 40 {
		t.Errorf("ctx size = %dx%d, want 120x40", got.ctx.Width, got.ctx.Height)
	}
	if _, ok := fakes[PageTopic].lastMsg.(tea.WindowSizeMsg); !ok {
		t.Errorf("size was not forwarded to the current page (last msg %T)", fakes[PageTopic].lastMsg)
	}
}

// A NavigateMsg must switch the page AND initialise it. Without the Init the page
// renders its zero state.
func TestNavigateSwitchesPageAndInitialisesIt(t *testing.T) {
	m, fakes := newShell(t, 80, 24)
	m.current = PageMenu

	next, cmd := m.Update(NavigateMsg{Page: PageAudio})
	got := next.(Model)

	if got.current != PageAudio {
		t.Fatalf("current = %v, want %v", got.current, PageAudio)
	}
	if cmd == nil {
		t.Fatal("NavigateMsg did not return the page's Init command")
	}
	if fakes[PageAudio].inits != 1 {
		t.Errorf("new page was initialised %d times, want 1", fakes[PageAudio].inits)
	}
	// And the command must be the NEW page's, not the one we navigated away from.
	im, ok := cmd().(initMsg)
	if !ok {
		t.Fatalf("Init command returned %T, want initMsg", cmd())
	}
	if want := PageAudio.String(); im.id != want {
		t.Errorf("Init command came from page %q, want %q", im.id, want)
	}
}

// Key messages the shell does not handle must still reach the page — otherwise
// every page's own bindings break silently.
func TestUnhandledKeysStillReachThePage(t *testing.T) {
	m, fakes := newShell(t, 80, 24)
	m.current = PageTopic

	next, _ := m.Update(tea.KeyPressMsg{Code: 'x', Text: "x"})
	if next.(Model).current != PageTopic {
		t.Errorf("an unhandled key changed the page to %v", next.(Model).current)
	}
	if _, ok := fakes[PageTopic].lastMsg.(tea.KeyPressMsg); !ok {
		t.Errorf("an unhandled key was not forwarded to the page (last msg %T)", fakes[PageTopic].lastMsg)
	}
}
