document.addEventListener("keydown", (event) => {
  if (event.key !== "Escape") {
    return;
  }

  const drawer = document.getElementById("__drawer");
  if (!(drawer instanceof HTMLInputElement) || !drawer.checked) {
    return;
  }

  event.preventDefault();
  drawer.checked = false;
  drawer.dispatchEvent(new Event("change", { bubbles: true }));
});
