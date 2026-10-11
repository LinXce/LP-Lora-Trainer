/** Preset switches are a view of the editable resolution text, not separate state.
 * Match the adapter's newline/comma, x/X/* and "width height" syntax.
 * Keep custom and invalid entries so a preset click never hides input errors.
 */
function entries(value: unknown): string[] {
  return typeof value === 'string' ? value.split(/[,\r\n]/).map(s => s.trim()).filter(Boolean) : []
}

function square(entry: string): number | null {
  const match = /^(\d+)x(\d+)$/.exec(entry.toLowerCase().replaceAll('*', 'x').replaceAll(' ', 'x'))
  if (!match) return null
  const width = Number(match[1]), height = Number(match[2])
  return width === height ? width : null
}

export function hasResolution(value: unknown, size: number): boolean {
  return entries(value).some(entry => square(entry) === size)
}

export function toggleResolution(value: unknown, size: number): string {
  const list = entries(value)
  return (hasResolution(value, size)
    ? list.filter(entry => square(entry) !== size)
    : [...list, `${size}x${size}`]).join('\n')
}
