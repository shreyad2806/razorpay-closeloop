"use client";

export function MobileToggle() {
  return (
    <button
      className="mobile-toggle"
      onClick={() => {
        document.getElementById("sidebar")?.classList.toggle("open");
        document.getElementById("overlay")?.classList.toggle("active");
      }}
      aria-label="Toggle sidebar"
    >
      ☰
    </button>
  );
}
