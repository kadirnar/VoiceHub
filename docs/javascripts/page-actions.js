(() => {
  const copyWithSelection = (text) => {
    const copyBuffer = document.createElement("textarea");
    copyBuffer.value = text;
    copyBuffer.setAttribute("readonly", "");
    copyBuffer.setAttribute("aria-hidden", "true");
    copyBuffer.style.position = "fixed";
    copyBuffer.style.inset = "0 auto auto -9999px";
    copyBuffer.style.opacity = "0";
    document.body.append(copyBuffer);
    copyBuffer.focus({ preventScroll: true });
    copyBuffer.select();
    const copied = document.execCommand("copy");
    copyBuffer.remove();
    if (!copied) throw new Error("The browser rejected both clipboard copy methods.");
  };

  const writeClipboard = async (text) => {
    try {
      await navigator.clipboard.writeText(text);
    } catch (_error) {
      copyWithSelection(text);
    }
  };

  const initializePageActions = () => {
    const button = document.querySelector("[data-vh-copy-page]");
    const article = button?.closest(".md-content__inner");
    const label = button?.querySelector("[data-vh-copy-page-label]");
    if (!(button instanceof HTMLButtonElement)
        || !(article instanceof HTMLElement)
        || !(label instanceof HTMLElement)) return;

    const originalLabel = label.textContent;
    let resetTimer;
    let copyInProgress = false;

    const copyPage = async ({ restoreKeyboardFocus = false } = {}) => {
      if (copyInProgress) return;
      copyInProgress = true;
      button.setAttribute("aria-busy", "true");

      const readablePage = article.cloneNode(true);
      readablePage.querySelectorAll(
        ".md-content__button, .headerlink, .md-clipboard, .md-source-file",
      ).forEach((element) => element.remove());

      window.clearTimeout(resetTimer);
      try {
        await writeClipboard(readablePage.innerText.trim());
        label.textContent = "Copied";
      } catch (_error) {
        label.textContent = "Copy failed";
      } finally {
        copyInProgress = false;
        button.setAttribute("aria-busy", "false");
        if (restoreKeyboardFocus) {
          button.focus({ preventScroll: true });
          button.classList.add("focus-visible");
        }
        resetTimer = window.setTimeout(() => {
          label.textContent = originalLabel;
        }, 1600);
      }
    };

    button.addEventListener("click", (event) => {
      const keyboardActivation = event.detail === 0;
      if (!keyboardActivation) button.classList.remove("focus-visible");
      void copyPage({ restoreKeyboardFocus: keyboardActivation });
    });
    button.addEventListener("keydown", (event) => {
      if (event.key !== "Enter" && event.key !== " ") return;
      event.preventDefault();
      button.classList.add("focus-visible");
      void copyPage({ restoreKeyboardFocus: true });
    });
    button.addEventListener("pointerdown", () => button.classList.remove("focus-visible"));
    button.addEventListener("blur", () => button.classList.remove("focus-visible"));
  };

  const initializeCodeActions = () => {
    document.querySelectorAll(".md-content__inner .highlight").forEach((container) => {
      const code = container.querySelector("pre > code");
      if (!(code instanceof HTMLElement)
          || container.querySelector("button[data-vh-code-copy]")) return;

      const button = document.createElement("button");
      button.type = "button";
      button.className = "md-clipboard md-icon";
      button.dataset.vhCodeCopy = "";
      button.title = "Copy to clipboard";
      button.setAttribute("aria-label", "Copy to clipboard");
      button.setAttribute("aria-busy", "false");
      container.append(button);

      let resetTimer;
      let copyInProgress = false;
      button.addEventListener("click", async (event) => {
        if (copyInProgress) return;
        const keyboardActivation = event.detail === 0;
        copyInProgress = true;
        button.setAttribute("aria-busy", "true");
        window.clearTimeout(resetTimer);
        try {
          await writeClipboard(code.textContent);
          button.classList.add("md-clipboard--active");
          button.title = "Copied to clipboard";
          button.setAttribute("aria-label", "Copied to clipboard");
        } catch (_error) {
          button.title = "Copy failed";
          button.setAttribute("aria-label", "Copy failed");
        } finally {
          copyInProgress = false;
          button.setAttribute("aria-busy", "false");
          if (keyboardActivation) {
            button.focus({ preventScroll: true });
            button.classList.add("focus-visible");
          }
          resetTimer = window.setTimeout(() => {
            button.classList.remove("md-clipboard--active");
            button.title = "Copy to clipboard";
            button.setAttribute("aria-label", "Copy to clipboard");
          }, 1600);
        }
      });
      button.addEventListener("pointerdown", () => button.classList.remove("focus-visible"));
      button.addEventListener("blur", () => button.classList.remove("focus-visible"));
    });
  };

  const initializeActions = () => {
    initializePageActions();
    initializeCodeActions();
  };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initializeActions, { once: true });
  } else {
    initializeActions();
  }
})();
