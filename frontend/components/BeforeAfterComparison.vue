<script setup lang="ts">
  interface Props {
    beforeImage: string
    afterImage: string
    beforeLabel?: string
    afterLabel?: string
    beforeAlt?: string
    afterAlt?: string
    initialPosition?: number
  }

  const props = withDefaults(defineProps<Props>(), {
    beforeLabel: 'Original',
    afterLabel: 'Generated',
    beforeAlt: 'Original image',
    afterAlt: 'Generated image',
    initialPosition: 50,
  })

  const containerRef = ref<HTMLElement>()
  const dividerPosition = ref(props.initialPosition)
  const isActive = ref(false)

  function handleImageLoad() {
    // Reset position when new images load
    dividerPosition.value = props.initialPosition
  }

  function handleMouseMove(event: MouseEvent) {
    if (!containerRef.value)
      return
    updatePosition(event.clientX)
  }

  function handleTouchMove(event: TouchEvent) {
    if (!containerRef.value)
      return
    event.preventDefault()
    const touch = event.touches[0]
    updatePosition(touch.clientX)
  }

  function handleMouseLeave() {
  // Optionally reset to center when mouse leaves
  // dividerPosition.value = 50
  }

  function updatePosition(clientX: number) {
    if (!containerRef.value)
      return

    const rect = containerRef.value.getBoundingClientRect()
    const x = clientX - rect.left
    const percentage = Math.max(0, Math.min(100, (x / rect.width) * 100))
    dividerPosition.value = percentage
  }
</script>

<template>
  <div ref="containerRef" class="relative w-full h-auto rounded-lg overflow-hidden bg-gray-900">
    <!-- Original image (underneath) -->
    <img
      :src="beforeImage"
      :alt="beforeAlt"
      class="w-full h-auto block"
      @load="handleImageLoad"
    />

    <!-- Generated image (on top, clipped) -->
    <div
      class="absolute inset-0 overflow-hidden"
      :style="{ clipPath: `inset(0 ${100 - dividerPosition}% 0 0)` }"
    >
      <img
        :src="afterImage"
        :alt="afterAlt"
        class="w-full h-auto block"
      />
    </div>

    <!-- Divider line and handle -->
    <div
      class="absolute top-0 bottom-0 w-1 bg-white shadow-lg z-10 flex items-center justify-center pointer-events-none"
      :style="{ left: `${dividerPosition}%` }"
    >
      <!-- Divider handle -->
      <div class="w-8 h-8 bg-white rounded-full shadow-lg flex items-center justify-center -ml-4">
        <div class="w-1 h-4 bg-gray-400 rounded-full mx-0.5"></div>
        <div class="w-1 h-4 bg-gray-400 rounded-full mx-0.5"></div>
      </div>
    </div>

    <!-- Invisible overlay for mouse tracking -->
    <div
      class="absolute inset-0 z-20 cursor-ew-resize"
      @mousemove="handleMouseMove"
      @touchmove="handleTouchMove"
      @mouseleave="handleMouseLeave"
    ></div>

    <!-- Labels -->
    <div class="absolute top-4 left-4 bg-black/50 text-white text-sm px-2 py-1 rounded">
      {{ beforeLabel }}
    </div>
    <div class="absolute top-4 right-4 bg-black/50 text-white text-sm px-2 py-1 rounded">
      {{ afterLabel }}
    </div>
  </div>
</template>

<style scoped>
.cursor-ew-resize {
  cursor: ew-resize;
}
</style>
