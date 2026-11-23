// Runs inside the built Vue app. Observe the real fetch responses, leaving requests untouched.
async function exerciseTrainingForm(test) {
  const calls = []
  const original = window.fetch.bind(window)
  window.fetch = async (url, init) => {
    const response = await original(url, init)
    const path = new URL(url, location.href).pathname
    if (/\/training\/(validate|submit)$/.test(path)) {
      calls.push({ path, body: JSON.parse(init.body), result: await response.clone().json() })
    }
    return response
  }
  const sleep = ms => new Promise(resolve => setTimeout(resolve, ms))
  async function wait(get, what) {
    const end = Date.now() + 10000
    while (Date.now() < end) {
      const value = get()
      if (value) return value
      await sleep(30)
    }
    throw Error(what + " — " + document.body.innerText.slice(-1000))
  }
  const buttons = () => [...document.querySelectorAll("button")]
  const button = text => buttons().find(b => b.textContent.trim() === text && !b.disabled)
  const base = label => [...document.querySelectorAll("label.field")].find(
    e => e.querySelector(".field__label")?.textContent.trim().startsWith(label))
  function edit(control, value) {
    control.focus()
    control.value = String(value)
    control.dispatchEvent(new Event("input", { bubbles: true }))
    control.dispatchEvent(new Event("change", { bubbles: true }))
    control.blur()
  }
  async function choose(field, value) {
    const option = await wait(() => [...(field.querySelector("select")?.options ?? [])].find(
      o => o.value === value && !o.disabled), "select " + value)
    const trigger = field.querySelector(".select-control__trigger")
    if (field.querySelector("select").value === value) return
    trigger.click()
    const popupOption = await wait(() => buttons().find(b => b.classList.contains("select-popup__option")
      && b.innerText.trim() === option.label && !b.disabled), "popup " + value)
    popupOption.click()
  }
  async function param(key) {
    const find = () => document.querySelector('[data-key="' + CSS.escape(key) + '"]')
    if (find()) return find()
    const search = await wait(() => document.querySelector(".param-search input"), "parameter search")
    edit(search, key)
    const hit = await wait(() => [...document.querySelectorAll(".param-search__hit")].find(
      e => e.querySelector("code")?.textContent === key), "search " + key)
    hit.click()
    return wait(find, "field " + key)
  }
  async function requestByClick(label, path) {
    const from = calls.length
    const b = await wait(() => button(label), "enabled button " + label)
    b.click()
    return wait(() => calls.slice(from).find(c => c.path.endsWith(path)), label)
  }
  if (test.engine === "anima_lora") {
    (await wait(() => button("MOSS-Image"), "MOSS architecture")).click()
    await choose(await wait(() => base("引擎安装实例"), "MOSS engine selector"), test.engine)
    const caption = await param("moss_caption_format")
    await choose(caption, "fixed")
    caption.querySelector('button[aria-label="字幕格式 详细说明"]').click()
    const drawer = await wait(() => document.querySelector('[role="dialog"][aria-label="字幕格式 详细说明"]'), "caption details")
    for (const format of ["text", "layout_json", "tags", "fixed"]) {
      if (!drawer.querySelector("pre").textContent.includes(format)) throw Error("missing description for " + format)
    }
    drawer.querySelector('button[aria-label="关闭"]').click()
    await wait(() => !document.body.contains(drawer), "close caption details")
    if (caption.querySelector("select").value !== "fixed") throw Error("opening details changed the caption format")
  }
  const architecture = await wait(() => button(test.archLabel), "architecture " + test.archLabel)
  architecture.click()
  await choose(await wait(() => base("引擎安装实例"), "engine selector"), test.engine)
  await wait(() => button("初始化参数") && document.querySelector(".param-search"), "capabilities")
  await choose(base("数据集"), test.dataset)
  edit(base("任务名称").querySelector("input"), test.engine)
  edit(base("输出根目录").querySelector("input"), test.output)
  if (base("底模文件")) edit(base("底模文件").querySelector("input"), test.model)
  const knob = () => param(test.knob).then(f => f.querySelector("input"))
  const input = await knob()
  const shown = Number(input.value)
  edit(input, -1)
  const rejected = await requestByClick("提交到队列", "/validate")
  if (rejected.body.params[test.knob] !== -1 || rejected.result.ok ||
      !rejected.result.issues.some(i => i.field === test.knob && i.level === "error"))
    throw Error("invalid parameter reached submit without a field error: " + test.knob)
  const issue = rejected.result.issues.find(i => i.field === test.knob && i.level === "error")
  await wait(() => document.querySelector(".preview__issues")?.textContent.includes(issue.message), "invalid field feedback")
  if (calls.some(c => c.path.endsWith("/submit"))) throw Error("invalid parameters were submitted")
  button("初始化参数").click()
  const confirm = await wait(() => button("确认初始化"), "reset confirmation")
  confirm.click()
  const after = Number((await knob()).value)
  const initialized = await requestByClick("检测参数", "/validate")
  for (const [key, value] of Object.entries(test.inputs)) {
    const field = await param(key)
    if (field.querySelector("select")) {
      await choose(field, String(value))
    } else if (field.querySelector("[role=switch]")) {
      if (field.querySelector("[role=switch]").getAttribute("aria-checked") !== String(value)) field.querySelector("button.bool").click()
    } else {
      edit(field.querySelector("input, textarea"), value)
    }
    await sleep(30)
  }
  const checked = await requestByClick("检测参数", "/validate")
  if (!checked.result.ok) throw Error("validation: " + JSON.stringify(checked.result.issues))
  const submitted = await requestByClick("提交到队列", "/submit")
  return { shown, after, initialized: initialized.body.params, checked, submitted }
}
