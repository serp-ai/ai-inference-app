<script setup lang="ts">
  interface Props {
    imageUrl: string
    open: boolean
    existingMask?: string // base64 mask data
  }

  const props = defineProps<Props>()

  const emit = defineEmits<{
    close: []
    saveMask: [string] // base64 mask data
  }>()

  // Canvas refs
  const canvasRef = ref<HTMLCanvasElement>()
  const overlayCanvasRef = ref<HTMLCanvasElement>()
  const maskCanvasRef = ref<HTMLCanvasElement>()

  // Drawing state
  const isDrawing = ref(false)
  const brushSize = ref(50)
  const maskMode = ref('paint') // 'paint' or 'erase'

  // Canvas contexts
  let ctx: CanvasRenderingContext2D | null = null
  let overlayCtx: CanvasRenderingContext2D | null = null
  let maskCtx: CanvasRenderingContext2D | null = null
  let imageElement: HTMLImageElement | null = null

  // Initialize canvas when modal opens
  watch(() => props.open, (isOpen) => {
    if (isOpen) {
      nextTick(() => {
        initializeCanvas()
      })
    }
  })

  function initializeCanvas() {
    if (!canvasRef.value || !overlayCanvasRef.value || !maskCanvasRef.value)
      return

    ctx = canvasRef.value.getContext('2d')
    overlayCtx = overlayCanvasRef.value.getContext('2d')
    maskCtx = maskCanvasRef.value.getContext('2d')

    if (!ctx || !overlayCtx || !maskCtx)
      return

    // Load and draw the image
    imageElement = new Image()
    imageElement.onload = () => {
      if (!ctx || !overlayCtx || !canvasRef.value || !overlayCanvasRef.value || !imageElement)
        return

      // Set canvas dimensions to match image
      const maxSize = 512 // Max canvas size for performance
      let { width, height } = imageElement

      if (width > maxSize || height > maxSize) {
        const ratio = Math.min(maxSize / width, maxSize / height)
        width = width * ratio
        height = height * ratio
      }

      canvasRef.value.width = width
      canvasRef.value.height = height
      overlayCanvasRef.value.width = width
      overlayCanvasRef.value.height = height
      maskCanvasRef.value.width = width
      maskCanvasRef.value.height = height

      // Draw the image on the main canvas
      ctx.drawImage(imageElement, 0, 0, width, height)

      // Initialize overlay with transparent background
      overlayCtx.clearRect(0, 0, width, height)

      // Initialize mask canvas with black background (no mask)
      maskCtx.fillStyle = 'black'
      maskCtx.fillRect(0, 0, width, height)

      // Load existing mask if provided
      if (props.existingMask) {
        loadExistingMask(width, height)
      }

      // Update overlay display
      updateOverlayDisplay()
    }

    imageElement.src = props.imageUrl
  }

  function startDrawing(event: MouseEvent) {
    if (!overlayCanvasRef.value || !overlayCtx)
      return

    isDrawing.value = true
    const rect = overlayCanvasRef.value.getBoundingClientRect()
    const x = event.clientX - rect.left
    const y = event.clientY - rect.top

    draw(x, y)
  }

  function draw(x: number, y: number) {
    if (!isDrawing.value || !maskCtx || !maskCanvasRef.value)
      return

    // Draw on the mask canvas (black/white)
    maskCtx.globalCompositeOperation = maskMode.value === 'paint' ? 'source-over' : 'destination-out'
    maskCtx.fillStyle = maskMode.value === 'paint' ? 'white' : 'black'
    maskCtx.beginPath()
    maskCtx.arc(x, y, brushSize.value / 2, 0, 2 * Math.PI)
    maskCtx.fill()

    // Update the visual overlay
    updateOverlayDisplay()
  }

  function onMouseMove(event: MouseEvent) {
    if (!isDrawing.value || !overlayCanvasRef.value)
      return

    const rect = overlayCanvasRef.value.getBoundingClientRect()
    const x = event.clientX - rect.left
    const y = event.clientY - rect.top

    draw(x, y)
  }

  function stopDrawing() {
    isDrawing.value = false
  }

  function clearMask() {
    if (!maskCtx || !maskCanvasRef.value || !overlayCtx || !overlayCanvasRef.value)
      return

    // Clear the mask canvas (set to black - no mask)
    maskCtx.fillStyle = 'black'
    maskCtx.fillRect(0, 0, maskCanvasRef.value.width, maskCanvasRef.value.height)

    // Clear the overlay display
    overlayCtx.clearRect(0, 0, overlayCanvasRef.value.width, overlayCanvasRef.value.height)
  }

  function updateOverlayDisplay() {
    if (!maskCtx || !overlayCtx || !maskCanvasRef.value || !overlayCanvasRef.value)
      return

    // Clear the overlay
    overlayCtx.clearRect(0, 0, overlayCanvasRef.value.width, overlayCanvasRef.value.height)

    // Get mask data and create red overlay for white areas
    const maskData = maskCtx.getImageData(0, 0, maskCanvasRef.value.width, maskCanvasRef.value.height)
    const overlayData = overlayCtx.createImageData(maskCanvasRef.value.width, maskCanvasRef.value.height)

    for (let i = 0; i < maskData.data.length; i += 4) {
      // If mask pixel is white, show red overlay
      if (maskData.data[i] > 128) { // White area in mask
        overlayData.data[i] = 255 // R
        overlayData.data[i + 1] = 0 // G
        overlayData.data[i + 2] = 0 // B
        overlayData.data[i + 3] = 77 // A (0.3 transparency)
      }
      else {
        overlayData.data[i + 3] = 0 // Transparent
      }
    }

    overlayCtx.putImageData(overlayData, 0, 0)
  }

  function saveMask() {
    if (!maskCanvasRef.value)
      return

    // The mask canvas already contains proper black/white data
    const base64Mask = maskCanvasRef.value.toDataURL('image/png').split(',')[1]
    emit('saveMask', base64Mask)
  }

  function loadExistingMask(canvasWidth: number, canvasHeight: number) {
    if (!props.existingMask || !maskCtx || !maskCanvasRef.value)
      return

    const maskImage = new Image()
    maskImage.onload = () => {
      if (!maskCtx || !maskCanvasRef.value)
        return

      // Draw the existing mask directly to the mask canvas
      maskCtx.drawImage(maskImage, 0, 0, canvasWidth, canvasHeight)

      // Update the overlay display
      updateOverlayDisplay()
    }

    maskImage.src = `data:image/png;base64,${props.existingMask}`
  }

  function handleClose() {
    emit('close')
  }
</script>

<template>
  <UModal
    :open="open"
    :ui="{ content: 'max-w-4xl' }"
    :dismissible="false"
    @close="handleClose"
  >
    <template #header>
      <div class="flex items-center gap-2">
        <UIcon name="i-lucide-edit" class="size-5 text-purple-400" />
        <span>Mask Editor</span>
      </div>
    </template>

    <template #body="{ close }">
      <div class="space-y-4">
        <!-- Controls -->
        <div class="flex flex-wrap items-center gap-4 p-4 bg-gray-800/50 rounded-lg">
          <!-- Mode Toggle -->
          <div class="flex items-center gap-2">
            <label class="text-sm font-medium text-white">Mode:</label>
            <div class="flex rounded-lg overflow-hidden border border-gray-600">
              <button
                class="px-3 py-1 text-xs font-medium transition-colors"
                :class="[
                  maskMode === 'paint'
                    ? 'bg-red-500 text-white'
                    : 'bg-gray-700 text-gray-300 hover:bg-gray-600',
                ]"
                @click="maskMode = 'paint'"
              >
                Paint Mask
              </button>
              <button
                class="px-3 py-1 text-xs font-medium transition-colors"
                :class="[
                  maskMode === 'erase'
                    ? 'bg-blue-500 text-white'
                    : 'bg-gray-700 text-gray-300 hover:bg-gray-600',
                ]"
                @click="maskMode = 'erase'"
              >
                Erase
              </button>
            </div>
          </div>

          <!-- Brush Size -->
          <div class="flex items-center gap-2">
            <label class="text-sm font-medium text-white">Brush:</label>
            <div class="flex items-center gap-2">
              <USlider
                v-model="brushSize"
                :min="5"
                :max="100"
                :step="1"
                class="w-20"
              />
              <span class="text-xs text-muted w-8">{{ brushSize }}</span>
            </div>
          </div>

          <!-- Clear Button -->
          <UButton
            color="neutral"
            variant="outline"
            size="xs"
            icon="i-lucide-trash-2"
            @click="clearMask"
          >
            Clear
          </UButton>
        </div>

        <!-- Canvas Container -->
        <div class="relative border border-gray-600 rounded-lg overflow-hidden bg-gray-900">
          <div class="relative inline-block">
            <!-- Main image canvas -->
            <canvas ref="canvasRef" class="block max-w-full h-auto"></canvas>

            <!-- Overlay canvas for mask visualization -->
            <canvas
              ref="overlayCanvasRef"
              class="absolute inset-0 cursor-crosshair"
              @mousedown="startDrawing"
              @mousemove="onMouseMove"
              @mouseup="stopDrawing"
              @mouseleave="stopDrawing"
            ></canvas>

            <!-- Hidden mask canvas for actual mask data -->
            <canvas ref="maskCanvasRef" class="hidden"></canvas>
          </div>
        </div>

        <!-- Instructions -->
        <div class="p-3 bg-blue-500/10 border border-blue-500/20 rounded-lg">
          <div class="flex items-start gap-2">
            <UIcon name="i-lucide-info" class="size-4 text-blue-400 mt-0.5 flex-shrink-0" />
            <div class="text-xs text-blue-400/90">
              <p><strong>Paint mode:</strong> Click and drag to mark areas for inpainting (shown in red)</p>
              <p><strong>Erase mode:</strong> Remove parts of the mask</p>
              <p>Only the red marked areas will be regenerated.</p>
            </div>
          </div>
        </div>
      </div>
    </template>

    <template #footer="{ close }">
      <div class="flex gap-3">
        <UButton color="neutral" variant="outline" @click="handleClose">
          Cancel
        </UButton>
        <UButton color="primary" icon="i-lucide-check" @click="saveMask">
          Apply Mask
        </UButton>
      </div>
    </template>
  </UModal>
</template>
