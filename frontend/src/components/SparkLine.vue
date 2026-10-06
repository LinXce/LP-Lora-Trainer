<script setup lang="ts">
/** Compact trend line for tiles: no axes, end dot only. Values stay in the adjacent text. */
import { computed } from 'vue'

const props = withDefaults(defineProps<{ values: number[]; width?: number; height?: number; dark?: boolean }>(), {
  width: 120,
  height: 32,
  dark: false,
})

const geo = computed(() => {
  const v = props.values
  if (v.length < 2) return null
  let lo = Infinity
  let hi = -Infinity
  for (const x of v) {
    lo = Math.min(lo, x)
    hi = Math.max(hi, x)
  }
  const span = hi - lo || 1
  const p = 4
  const xs = (i: number) => p + (i / (v.length - 1)) * (props.width - p * 2)
  const ys = (x: number) => p + (1 - (x - lo) / span) * (props.height - p * 2)
  const d = v.map((x, i) => `${i ? 'L' : 'M'}${xs(i).toFixed(1)},${ys(x).toFixed(1)}`).join('')
  return { d, ex: xs(v.length - 1), ey: ys(v[v.length - 1]) }
})
</script>

<template>
  <svg :width="width" :height="height" class="spark" :class="{ 'spark--dark': dark }" aria-hidden="true">
    <template v-if="geo">
      <path :d="geo.d" class="spark__line" />
      <circle :cx="geo.ex" :cy="geo.ey" r="3" class="spark__dot" />
    </template>
    <line v-else x1="4" :x2="width - 4" :y1="height / 2" :y2="height / 2" class="spark__none" />
  </svg>
</template>

<style scoped>
.spark {
  display: block;
  overflow: visible;
}
.spark__line {
  fill: none;
  stroke: var(--light-strong);
  stroke-width: 2;
  stroke-linejoin: round;
  stroke-linecap: round;
}
.spark__dot {
  fill: var(--light-strong);
}
.spark__none {
  stroke: rgba(255, 255, 255, 0.2);
  stroke-dasharray: 3 4;
}
.spark--dark .spark__line {
  stroke: var(--ink);
}
.spark--dark .spark__dot {
  fill: var(--ink);
}
.spark--dark .spark__none {
  stroke: rgba(0, 0, 0, 0.25);
}
</style>
