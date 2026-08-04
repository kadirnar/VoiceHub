(() => {
  const drawer = document.getElementById("__drawer");
  const trigger = document.querySelector("[data-vh-drawer-trigger]");
  const navigation = document.querySelector(".md-sidebar--primary");
  if (!(drawer instanceof HTMLInputElement) || !(trigger instanceof HTMLButtonElement)
      || !(navigation instanceof HTMLElement)) return;

  const mobileViewport = window.matchMedia("(max-width: 59.984375em)");
  if (!navigation.id) navigation.id = "__primary_navigation";
  trigger.setAttribute("aria-controls", navigation.id);

  let focusNavigationAfterOpen = false;
  let restoreTriggerAfterClose = false;
  const synchronizeDrawerState = () => {
    const expanded = mobileViewport.matches && drawer.checked;
    navigation.inert = mobileViewport.matches && !drawer.checked;
    trigger.setAttribute("aria-expanded", String(expanded));

    if (expanded && focusNavigationAfterOpen) {
      focusNavigationAfterOpen = false;
      requestAnimationFrame(() => requestAnimationFrame(() => {
        const firstFocusTarget = navigation.querySelector(
          "a.md-nav__button[href], button.md-nav__link[data-vh-nav-toggle], a.md-nav__link[href]"
        );
        if (firstFocusTarget instanceof HTMLElement) firstFocusTarget.focus();
      }));
    } else if (!drawer.checked && restoreTriggerAfterClose) {
      restoreTriggerAfterClose = false;
      requestAnimationFrame(() => requestAnimationFrame(() => trigger.focus()));
    }
  };
  const setDrawerState = (expanded) => {
    drawer.checked = expanded;
    drawer.dispatchEvent(new Event("change", { bubbles: true }));
  };

  trigger.addEventListener("click", (event) => {
    event.preventDefault();
    event.stopImmediatePropagation();
    focusNavigationAfterOpen = !drawer.checked;
    restoreTriggerAfterClose = drawer.checked;
    setDrawerState(!drawer.checked);
  }, true);
  drawer.addEventListener("change", synchronizeDrawerState);
  mobileViewport.addEventListener("change", synchronizeDrawerState);
  document.addEventListener("keydown", (event) => {
    if (event.key !== "Escape" || !mobileViewport.matches || !drawer.checked) return;
    event.preventDefault();
    event.stopImmediatePropagation();
    restoreTriggerAfterClose = true;
    setDrawerState(false);
  }, true);
  synchronizeDrawerState();
})();
