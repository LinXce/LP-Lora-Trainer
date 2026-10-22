<script setup lang="ts">
/* Dumb empty placeholder; the optional illustration can be a clickable mascot
   (e.g. the overview character) that reacts with a quick shake. */
import { ref } from 'vue'
import AppIcon from './AppIcon.vue'

const props = defineProps<{
  icon?: string
  image?: string
  artSize?: number
  title: string
  text?: string
  /** Render the illustration as a button that emits `image-click`. */
  imageClickable?: boolean
}>()
const emit = defineEmits<{
  (event: 'image-click'): void
  /** The click reaction (shake) finished; callers that swapped the image revert now. */
  (event: 'image-reaction-end'): void
}>()
const artRef = ref<HTMLElement | null>(null)
/* Latest reaction id: cancelling an older animation must not report its end. */
let reaction = 0

/* The whole illustration tile shakes, not just the picture: the light rounded
   background is part of the mascot and must move with it. Web Animations is used
   instead of a CSS class so every click restarts from the first frame. */
function poke() {
  emit('image-click')
  const element = artRef.value
  if (!props.imageClickable || !element) return
  reaction += 1
  const current = reaction
  const finished = () => {
    if (current === reaction) emit('image-reaction-end')
  }
  // Reduced motion keeps the reaction as a short static flash instead of motion.
  if (typeof element.animate !== 'function' || window.matchMedia?.('(prefers-reduced-motion: reduce)').matches) {
    window.setTimeout(finished, 520)
    return
  }
  const frames = [
    { transform: 'translate3d(0, 0, 0) rotate(0deg)' },
    { transform: 'translate3d(-5px, 0, 0) rotate(-3deg)' },
    { transform: 'translate3d(5px, 0, 0) rotate(3deg)' },
    { transform: 'translate3d(-4px, 0, 0) rotate(-2deg)' },
    { transform: 'translate3d(3px, 0, 0) rotate(1.5deg)' },
    { transform: 'translate3d(-2px, 0, 0) rotate(-0.8deg)' },
    { transform: 'translate3d(0, 0, 0) rotate(0deg)' },
  ]
  element.getAnimations().forEach((animation) => animation.cancel())
  const animation = element.animate(frames, { duration: 460, easing: 'ease-in-out' })
  // A cancelled (superseded) animation rejects here; only the latest reports its end.
  animation.finished.then(finished).catch(() => {})
}
</script>

<template>
  <div class="empty">
    <div ref="artRef" class="empty__art" :style="artSize ? { '--empty-art-size': `${artSize}px` } : undefined">
      <button
        v-if="image && imageClickable"
        type="button"
        class="empty__art-hit"
        title="点一下"
        aria-label="点击角色"
        @click="poke"
      >
        <img :src="image" alt="" />
      </button>
      <img v-else-if="image" :src="image" alt="" />
      <AppIcon v-else :name="icon ?? 'image'" :size="34" />
    </div>
    <p class="title-md">{{ title }}</p>
    <p v-if="text" class="empty__text">{{ text }}</p>
    <div class="empty__actions"><slot /></div>
  </div>
</template>

<style scoped>
.empty {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 10px;
  padding: 40px 24px;
  text-align: center;
  height: 100%;
}
.empty__art {
  display: grid;
  place-items: center;
  width: var(--empty-art-size, 76px);
  height: var(--empty-art-size, 76px);
  border-radius: 24px;
  background: rgba(255, 255, 255, 0.06);
  color: var(--light);
  margin-bottom: 6px;
  will-change: transform;
}
.empty__art img {
  width: 100%;
  height: 100%;
  border-radius: inherit;
  object-fit: cover;
}
/* Interactive illustration: a plain button surface, so click and keyboard focus
   stay native while the tile keeps its rounded shape. */
.empty__art-hit {
  display: grid;
  place-items: center;
  width: 100%;
  height: 100%;
  padding: 0;
  border: 0;
  border-radius: inherit;
  background: transparent;
  color: inherit;
  cursor: pointer;
}
.empty__art-hit:focus-visible {
  outline: 2px solid var(--focus);
  outline-offset: 3px;
}
.empty__art-hit:hover img {
  filter: brightness(1.05);
}
.empty__text {
  max-width: 380px;
  color: var(--text-3);
  font-size: 12px;
}
.empty__actions {
  display: flex;
  gap: 8px;
  margin-top: 6px;
}
.empty__actions:empty {
  display: none;
}
</style>
