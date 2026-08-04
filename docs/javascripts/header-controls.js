(() => {
  const initializeShellScrollTracking = () => {
    const header = document.querySelector(".md-header");
    if (!(header instanceof HTMLElement)) return;

    let updateScheduled = false;
    const updateShellScrollOffset = () => {
      updateScheduled = false;
      const offset = Math.min(window.scrollY, header.offsetHeight);
      document.documentElement.style.setProperty("--vh-shell-scroll-offset", `${offset}px`);
    };
    const scheduleShellScrollUpdate = () => {
      if (updateScheduled) return;
      updateScheduled = true;
      requestAnimationFrame(updateShellScrollOffset);
    };

    window.addEventListener("scroll", scheduleShellScrollUpdate, { passive: true });
    window.addEventListener("resize", scheduleShellScrollUpdate, { passive: true });
    updateShellScrollOffset();
  };

  const initializePrimaryNavigationControl = () => {
    const navigation = document.querySelector(".md-sidebar--primary");
    if (!(navigation instanceof HTMLElement)) return;

    navigation.querySelectorAll("label.md-nav__link[for]").forEach((label, index) => {
      if (!(label instanceof HTMLLabelElement)) return;
      const toggle = document.getElementById(label.htmlFor);
      const panel = Array.from(toggle?.parentElement?.children || [])
        .find((element) => element.classList.contains("md-nav"));
      if (!(toggle instanceof HTMLInputElement) || !(panel instanceof HTMLElement)) return;

      if (!panel.id) panel.id = `${toggle.id}_panel`;
      const button = document.createElement("button");
      Array.from(label.attributes).forEach((attribute) => {
        if (attribute.name !== "for" && attribute.name !== "role" && attribute.name !== "tabindex") {
          button.setAttribute(attribute.name, attribute.value);
        }
      });
      button.type = "button";
      button.innerHTML = label.innerHTML;
      button.dataset.vhNavToggle = toggle.id;
      button.setAttribute("aria-controls", panel.id);
      label.replaceWith(button);
      const sectionName = button.textContent?.trim().replace(/\s+/g, " ") || "Untitled";
      panel.setAttribute("aria-label", `Navigation section ${index + 1}: ${sectionName}`);
      panel.removeAttribute("aria-labelledby");
      if (panel.classList.contains("md-nav--secondary")) {
        panel.querySelectorAll("nav.md-nav").forEach((nestedPanel, nestedIndex) => {
          if (!(nestedPanel instanceof HTMLElement)) return;
          const labelledBy = nestedPanel.getAttribute("aria-labelledby");
          const labelledElement = labelledBy ? document.getElementById(labelledBy) : null;
          const nestedName = nestedPanel.getAttribute("aria-label")
            || labelledElement?.textContent?.trim().replace(/\s+/g, " ")
            || "Untitled";
          nestedPanel.setAttribute(
            "aria-label",
            `${sectionName} subsection ${nestedIndex + 1}: ${nestedName}`,
          );
          nestedPanel.removeAttribute("aria-labelledby");
        });
      }
      const synchronizeExpandedState = () => {
        button.setAttribute("aria-expanded", String(toggle.checked));
      };

      toggle.addEventListener("change", synchronizeExpandedState);
      button.addEventListener("click", (event) => {
        event.preventDefault();
        event.stopImmediatePropagation();
        toggle.checked = !toggle.checked;
        toggle.dispatchEvent(new Event("change", { bubbles: true }));
      }, true);
      synchronizeExpandedState();
    });
  };

  const initializeTableOfContentsTracking = () => {
    const tableOfContents = document.querySelector(".md-sidebar--secondary");
    if (!(tableOfContents instanceof HTMLElement)) return;

    tableOfContents.addEventListener("click", (event) => {
      const link = event.target instanceof Element
        ? event.target.closest('.md-nav__link[href^="#"]')
        : null;
      if (!(link instanceof HTMLAnchorElement) || !link.hash) return;

      let settleTimer;
      const preserveSettledAnchor = () => {
        if (window.location.hash !== link.hash) {
          window.history.replaceState(window.history.state, "", link.hash);
        }
        tableOfContents.querySelectorAll('.md-nav__link[href^="#"]').forEach((trackedLink) => {
          trackedLink.classList.toggle("md-nav__link--active", trackedLink === link);
        });
      };
      const scheduleSettledAnchor = () => {
        window.clearTimeout(settleTimer);
        settleTimer = window.setTimeout(() => {
          window.removeEventListener("scroll", scheduleSettledAnchor);
          preserveSettledAnchor();
          window.setTimeout(preserveSettledAnchor, 500);
        }, 150);
      };

      window.addEventListener("scroll", scheduleSettledAnchor, { passive: true });
      scheduleSettledAnchor();
    });
    tableOfContents.addEventListener("keydown", (event) => {
      if (event.key !== "Enter" || event.altKey || event.ctrlKey || event.metaKey || event.shiftKey) return;
      const link = event.target instanceof Element
        ? event.target.closest('.md-nav__link[href^="#"]')
        : null;
      if (!(link instanceof HTMLAnchorElement) || !link.hash) return;
      event.preventDefault();
      link.click();
      link.focus({ preventScroll: true });
    });
  };

  const initializeVersionControl = () => {
    const control = document.querySelector("[data-vh-version-control]");
    if (!(control instanceof HTMLDetailsElement)) return;

    const summary = control.querySelector("summary");
    if (!(summary instanceof HTMLElement)) return;

    const synchronizeExpandedState = () => {
      summary.setAttribute("aria-expanded", String(control.open));
    };

    control.addEventListener("toggle", synchronizeExpandedState);
    control.addEventListener("keydown", (event) => {
      if (event.target === summary && (event.key === "Enter" || event.key === " ")) {
        event.preventDefault();
        control.open = !control.open;
        return;
      }
      if (event.key !== "Escape" || !control.open) return;
      control.open = false;
      summary.focus();
    });
    document.addEventListener("pointerdown", (event) => {
      if (control.open && !control.contains(event.target)) control.open = false;
    });
    synchronizeExpandedState();
  };

  const initializeSearchControl = () => {
    const checkbox = document.querySelector("#__search");
    const trigger = document.querySelector("[data-vh-search-trigger]");
    const search = document.querySelector(".md-search");
    const input = search?.querySelector(".md-search__input");
    const output = search?.querySelector(".md-search__output");
    const scrollwrap = output?.querySelector(".md-search__scrollwrap");
    const primaryShortcut = search?.querySelector("[data-vh-search-shortcut-primary]");
    const mobileViewport = window.matchMedia("(max-width: 59.984375em)");
    if (!(checkbox instanceof HTMLInputElement) || !(trigger instanceof HTMLElement)
        || !(search instanceof HTMLElement) || !(input instanceof HTMLInputElement)
        || !(output instanceof HTMLElement) || !(scrollwrap instanceof HTMLElement)) return;

    if (primaryShortcut instanceof HTMLElement) {
      const applePlatform = /Mac|iPhone|iPad/.test(navigator.platform);
      primaryShortcut.textContent = applePlatform ? "⌘K" : "Ctrl K";
    }

    let restoreFocus = false;
    const focusClosedSearchTarget = () => {
      const closeFocusTarget = mobileViewport.matches ? trigger : input;
      requestAnimationFrame(() => requestAnimationFrame(() => {
        if (!restoreFocus) return;
        closeFocusTarget.focus();
      }));
    };
    const synchronizeExpandedState = () => {
      const expanded = checkbox.checked;
      trigger.setAttribute("aria-expanded", String(expanded));
      document.body.classList.toggle("vh-search-open", expanded);
      input.tabIndex = expanded || !mobileViewport.matches ? 0 : -1;
      output.setAttribute("aria-hidden", String(!expanded));
      output.inert = !expanded;
      scrollwrap.tabIndex = expanded ? 0 : -1;
      if (expanded) {
        restoreFocus = true;
      } else if (restoreFocus) {
        focusClosedSearchTarget();
      }
    };
    const openSearch = () => {
      restoreFocus = true;
      if (!checkbox.checked) {
        checkbox.checked = true;
        checkbox.dispatchEvent(new Event("change", { bubbles: true }));
      }
      input.focus({ preventScroll: true });
    };
    const closeSearch = () => {
      restoreFocus = true;
      if (checkbox.checked) {
        checkbox.checked = false;
        checkbox.dispatchEvent(new Event("change", { bubbles: true }));
      } else {
        focusClosedSearchTarget();
      }
    };

    checkbox.addEventListener("change", synchronizeExpandedState);
    trigger.addEventListener("click", (event) => {
      event.preventDefault();
      event.stopImmediatePropagation();
      openSearch();
    }, true);
    mobileViewport.addEventListener("change", synchronizeExpandedState);
    document.addEventListener("keydown", (event) => {
      if (event.key === "Tab" && document.activeElement === input) {
        restoreFocus = false;
        return;
      }
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") {
        event.preventDefault();
        openSearch();
        return;
      }
      if (event.key === "Escape" && checkbox.checked) {
        event.preventDefault();
        event.stopImmediatePropagation();
        closeSearch();
      }
    }, true);
    synchronizeExpandedState();
  };

  const initializeLanguageControl = () => {
    const select = document.querySelector("[data-vh-language-select]");
    if (!(select instanceof HTMLSelectElement)) return;

    const paletteTransferKey = "vh-language-palette-transfer";
    try {
      const pendingPalette = sessionStorage.getItem(paletteTransferKey);
      sessionStorage.removeItem(paletteTransferKey);
      const pendingInput = document.querySelector(
        `input[data-md-color-scheme="${pendingPalette || ""}"]`
      );
      if (pendingInput instanceof HTMLInputElement && !pendingInput.checked) {
        pendingInput.checked = true;
        pendingInput.dispatchEvent(new Event("change", { bubbles: true }));
      }
    } catch (error) {
      console.debug("Language palette transfer is unavailable", error);
    }

    select.addEventListener("keydown", (event) => {
      if (event.altKey || event.ctrlKey || event.metaKey || event.shiftKey) return;
      const direction = event.key === "ArrowDown" ? 1 : event.key === "ArrowUp" ? -1 : 0;
      if (!direction) return;
      const nextIndex = Math.max(0, Math.min(select.options.length - 1, select.selectedIndex + direction));
      if (nextIndex === select.selectedIndex) return;
      event.preventDefault();
      select.selectedIndex = nextIndex;
      select.dispatchEvent(new Event("change", { bubbles: true }));
    });
    select.addEventListener("change", () => {
      try {
        sessionStorage.setItem(paletteTransferKey, document.body.dataset.mdColorScheme || "default");
      } catch (error) {
        console.debug("Language palette transfer is unavailable", error);
      }
      if (select.value) window.location.assign(select.value);
    });
  };

  const initializeThemeControl = () => {
    const control = document.querySelector('[data-vh-header-control="theme"]');
    if (!(control instanceof HTMLElement)) return;

    const focusVisibleToggle = () => {
      requestAnimationFrame(() => requestAnimationFrame(() => {
        const visibleToggle = control.querySelector("[data-vh-theme-toggle]:not([hidden])");
        if (visibleToggle instanceof HTMLElement) visibleToggle.focus();
      }));
    };
    control.addEventListener("click", (event) => {
      const toggle = event.target instanceof Element
        ? event.target.closest("[data-vh-theme-toggle]")
        : null;
      if (!(toggle instanceof HTMLButtonElement)) return;
      const target = document.getElementById(toggle.dataset.vhThemeTarget || "");
      if (!(target instanceof HTMLInputElement)) return;
      target.checked = true;
      target.dispatchEvent(new Event("change", { bubbles: true }));
      focusVisibleToggle();
    });
    control.addEventListener("change", focusVisibleToggle);
  };

  const initializeCodeBlockLandmarks = () => {
    document.querySelectorAll(".md-code__nav").forEach((nav, index) => {
      if (!(nav instanceof HTMLElement)) return;
      nav.setAttribute("aria-label", `Code block ${index + 1} actions`);
    });
  };

  const initializeScrollableRegions = () => {
    const normalizeScrollableRegions = () => {
      document.querySelectorAll(".md-typeset__table").forEach((region, index) => {
        if (!(region instanceof HTMLElement)) return;
        region.tabIndex = 0;
        region.setAttribute("aria-label", `Scrollable table ${index + 1}`);
      });
      document.querySelectorAll(".md-typeset pre > code").forEach((region, index) => {
        if (!(region instanceof HTMLElement)) return;
        const scrollable = region.clientWidth > 0 && region.scrollWidth > region.clientWidth + 1;
        if (scrollable) {
          region.tabIndex = 0;
          region.dataset.vhScrollableCode = "true";
          region.setAttribute("aria-label", `Scrollable code block ${index + 1}`);
        } else if (region.dataset.vhScrollableCode === "true") {
          region.removeAttribute("tabindex");
          region.removeAttribute("aria-label");
          delete region.dataset.vhScrollableCode;
        }
      });
      document.querySelectorAll(".md-typeset .tabbed-labels").forEach((region, index) => {
        if (!(region instanceof HTMLElement)) return;
        const scrollable = region.clientWidth > 0 && region.scrollWidth > region.clientWidth + 1;
        if (scrollable) {
          region.tabIndex = 0;
          region.dataset.vhScrollableTabs = "true";
          region.setAttribute("aria-label", `Scrollable options ${index + 1}`);
        } else if (region.dataset.vhScrollableTabs === "true") {
          region.removeAttribute("tabindex");
          region.removeAttribute("aria-label");
          delete region.dataset.vhScrollableTabs;
        }
      });
      document.querySelectorAll(".md-search-result article pre > code").forEach((region, index) => {
        if (!(region instanceof HTMLElement)) return;
        region.tabIndex = 0;
        region.setAttribute("aria-label", `Search result code ${index + 1}`);
      });
    };

    normalizeScrollableRegions();
    const searchResult = document.querySelector(".md-search-result");
    if (searchResult instanceof HTMLElement) {
      new MutationObserver(normalizeScrollableRegions).observe(searchResult, {
        childList: true,
        subtree: true,
      });
    }
    document.addEventListener("change", (event) => {
      if (!(event.target instanceof HTMLInputElement)
          || !event.target.matches('.tabbed-set > input[type="radio"]')) return;
      requestAnimationFrame(normalizeScrollableRegions);
    });
    window.addEventListener("resize", normalizeScrollableRegions);
  };

  const initializeContentTabFocus = () => {
    document.querySelectorAll('.tabbed-set > input[type="radio"][id]').forEach((input) => {
      if (!(input instanceof HTMLInputElement) || !(input.parentElement instanceof HTMLElement)) return;
      const label = Array.from(
        input.parentElement.querySelectorAll(".tabbed-labels > label[for]"),
      ).find((candidate) => candidate instanceof HTMLLabelElement && candidate.htmlFor === input.id);
      if (!(label instanceof HTMLLabelElement)) return;

      input.addEventListener("focus", () => {
        label.scrollIntoView({ block: "nearest", inline: "nearest" });
        label.classList.add("vh-content-tab--focus");
      });
      input.addEventListener("blur", () => {
        label.classList.remove("vh-content-tab--focus");
      });
    });
  };

  const initializeSequentialFocusBoundary = () => {
    const skipLink = document.querySelector(".md-skip");
    if (!(skipLink instanceof HTMLAnchorElement)) return;

    document.addEventListener("keydown", (event) => {
      if (event.key !== "Tab" || event.shiftKey || event.altKey || event.ctrlKey || event.metaKey
          || document.activeElement !== document.body) return;
      event.preventDefault();
      skipLink.focus();
    }, true);
  };

  const initializeSourceControl = () => {
    const link = document.querySelector('[data-vh-header-control="source"] a[href]');
    if (!(link instanceof HTMLAnchorElement)) return;

    link.addEventListener("keydown", (event) => {
      if (event.key !== "Enter" || event.altKey || event.ctrlKey || event.metaKey || event.shiftKey) return;
      event.preventDefault();
      window.location.assign(link.href);
    });
  };

  const initializeHeaderControls = () => {
    initializeShellScrollTracking();
    initializePrimaryNavigationControl();
    initializeTableOfContentsTracking();
    initializeVersionControl();
    initializeSearchControl();
    initializeLanguageControl();
    initializeThemeControl();
    initializeCodeBlockLandmarks();
    initializeScrollableRegions();
    initializeContentTabFocus();
    initializeSequentialFocusBoundary();
    initializeSourceControl();
  };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initializeHeaderControls, { once: true });
  } else {
    initializeHeaderControls();
  }
})();
