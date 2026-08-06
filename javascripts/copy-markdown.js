(() => {
  const originalLabel = "Copy as Markdown";
  const copiedLabel = "Markdown copied";
  const failedLabel = "Could not copy Markdown";

  const sourceUrl = (button) => {
    const editUrl = new URL(button.dataset.mdCopyMarkdownSource);
    const marker = "/docs/";
    const markerIndex = editUrl.pathname.indexOf(marker);
    const script = document.querySelector('script[src$="javascripts/copy-markdown.js"]');
    if (markerIndex < 0 || !(script instanceof HTMLScriptElement)) {
      throw new Error("The Markdown source URL could not be resolved.");
    }
    const sourcePath = editUrl.pathname
      .slice(markerIndex + marker.length)
      .replace(/\.md$/, ".txt");
    return new URL(`../_markdown/${sourcePath}`, script.src);
  };

  const writeToClipboard = async (text) => {
    if (navigator.clipboard && window.isSecureContext) {
      await navigator.clipboard.writeText(text);
      return;
    }

    const textarea = document.createElement("textarea");
    textarea.value = text;
    textarea.style.position = "fixed";
    textarea.style.opacity = "0";
    document.body.appendChild(textarea);
    textarea.select();
    const copied = document.execCommand("copy");
    textarea.remove();
    if (!copied) {
      throw new Error("The browser rejected the clipboard operation.");
    }
  };

  const announce = (message) => {
    let region = document.querySelector("[data-md-copy-markdown-status]");
    if (!region) {
      region = document.createElement("span");
      region.className = "md-copy-markdown__status";
      region.dataset.mdCopyMarkdownStatus = "";
      region.setAttribute("role", "status");
      region.setAttribute("aria-live", "polite");
      document.body.appendChild(region);
    }
    region.textContent = message;
  };

  const initialize = () => {
    document.querySelectorAll("[data-md-copy-markdown]").forEach((button) => {
      if (button.dataset.mdCopyMarkdownReady !== undefined) {
        return;
      }
      button.dataset.mdCopyMarkdownReady = "";
      button.addEventListener("click", async () => {
        button.disabled = true;
        try {
          const response = await fetch(sourceUrl(button));
          if (!response.ok) {
            throw new Error(`Markdown source returned HTTP ${response.status}.`);
          }
          await writeToClipboard(await response.text());
          button.dataset.mdCopyMarkdownState = "copied";
          button.title = copiedLabel;
          button.setAttribute("aria-label", copiedLabel);
          announce(copiedLabel);
        } catch {
          button.dataset.mdCopyMarkdownState = "failed";
          button.title = failedLabel;
          button.setAttribute("aria-label", failedLabel);
          announce(failedLabel);
        } finally {
          window.setTimeout(() => {
            delete button.dataset.mdCopyMarkdownState;
            button.title = originalLabel;
            button.setAttribute("aria-label", originalLabel);
            button.disabled = false;
          }, 2000);
        }
      });
    });
  };

  if (typeof document$ !== "undefined") {
    document$.subscribe(initialize);
  } else if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initialize, { once: true });
  } else {
    initialize();
  }
})();
