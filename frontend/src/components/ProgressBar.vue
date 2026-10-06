<script setup lang="ts">
/** Segmented progress bar; renders an "unknown" striped track when ratio is null. */
const props = withDefaults(defineProps<{ ratio: number | null; segments?: number; light?: boolean }>(), {
  segments: 0,
  light: false,
})
</script>

<template>
  <div
    class="bar"
    :class="{ 'bar--unknown': props.ratio === null, 'bar--light': props.light }"
    role="progressbar"
    :aria-valuenow="props.ratio === null ? undefined : Math.round(props.ratio * 100)"
    aria-valuemin="0"
    aria-valuemax="100"
  >
    <template v-if="props.segments > 0">
      <span
        v-for="i in props.segments"
        :key="i"
        class="bar__seg"
        :class="{ 'is-on': props.ratio !== null && i / props.segments <= props.ratio + 1e-9 }"
      />
    </template>
    <span v-else-if="props.ratio !== null" class="bar__fill" :style="{ width: `${props.ratio * 100}%` }" />
  </div>
</template>

<style scoped>
.bar {
  position: relative;
  display: flex;
  gap: 4px;
  height: 6px;
  border-radius: var(--radius-pill);
  background: rgba(255, 255, 255, 0.08);
  overflow: hidden;
}
.bar--light {
  background: rgba(0, 0, 0, 0.12);
}
.bar__fill {
  height: 100%;
  border-radius: inherit;
  background: var(--light);
  transition: width 0.6s var(--ease);
}
.bar--light .bar__fill {
  background: var(--ink);
}
.bar__seg {
  flex: 1;
  height: 100%;
  border-radius: var(--radius-pill);
  background: rgba(255, 255, 255, 0.12);
  transition: background 0.3s;
}
.bar__seg.is-on {
  background: var(--light);
}
.bar--unknown {
  background: repeating-linear-gradient(
    -45deg,
    rgba(255, 255, 255, 0.05) 0 6px,
    rgba(255, 255, 255, 0.12) 6px 12px
  );
}
</style>
