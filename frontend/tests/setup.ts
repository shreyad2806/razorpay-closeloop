import "@testing-library/jest-dom";

/**
 * jsdom does not implement a few browser APIs used by the dashboard shell.
 * Stub them so component tests can render without touching a real browser.
 */
if (typeof window !== "undefined") {
  if (typeof window.matchMedia !== "function") {
    window.matchMedia = (query: string): MediaQueryList =>
      ({
        matches: false,
        media: query,
        onchange: null,
        addListener: () => undefined,
        removeListener: () => undefined,
        addEventListener: () => undefined,
        removeEventListener: () => undefined,
        dispatchEvent: () => false,
      }) as unknown as MediaQueryList;
  }

  if (typeof window.ResizeObserver !== "function") {
    window.ResizeObserver = class ResizeObserver {
      observe() {}
      unobserve() {}
      disconnect() {}
    } as unknown as typeof ResizeObserver;
  }

  if (typeof window.scrollTo !== "function") {
    window.scrollTo = (() => undefined) as unknown as typeof window.scrollTo;
  }
}
