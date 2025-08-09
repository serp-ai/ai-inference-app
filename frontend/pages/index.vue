<script setup lang="ts">
useHead({
  title: 'NSFW Clothes Remover',
  meta: [
    { name: 'description', content: 'Remove clothes from AI characters' },
  ],
})

const toast = useToast()

const steps = ref(25)
const guidance = ref(2.5)
const seed = ref()
const src = ref('')
const loading = ref(false)
const generationHistory = ref([])
const generatedImages = ref(null)
const config = useRuntimeConfig()

const uploadedImage = ref(null)
const uploadedImageUrl = ref('')
const maskData = ref(null)
const strength = ref(0.97)

// Modal state for mask editor
const showMaskEditor = ref(false)

// Download image function
function downloadImage(image) {
  const link = document.createElement('a')
  link.href = image.url
  link.download = `generated-${Date.now()}.png`
  link.click()
}

async function generateImage() {
  if (loading.value)
    return

  // Validate requirements for different modes
  if (!uploadedImage.value) {
    toast.add({
      title: 'Image Required',
      description: `Please upload an image`,
      color: 'red',
    })
    return
  }

  loading.value = true

  try {
    // Send original image and mask separately
    const inputImageB64 = await fileToBase64(uploadedImage.value)

    const requestBody: any = {
      input_image: inputImageB64,
      num_steps: steps.value,
      guidance: guidance.value,
      strength: strength.value,
      seed: seed.value ? Number.parseInt(seed.value) : -1,
      resize_dimension: 1152,
      do_resize: true,
    }

    // Add mask if exists
    if (maskData.value) {
      requestBody.mask_image = maskData.value
    }

    const response = await $fetch(`${config.public.apiBase}/api/inpainting/generate-clothes-removal`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: requestBody
    })

    // Store all generated images
    generatedImages.value = {
      composite_image: `data:image/png;base64,${response.composite_image}`,
      gen_image: `data:image/png;base64,${response.gen_image}`,
      input_image: `data:image/png;base64,${response.input_image}`,
      mask: `data:image/png;base64,${response.mask}`,
    }

    const newImage = {
      src: generatedImages.value.composite_image,
      originalSrc: generatedImages.value.input_image,
      steps: steps.value,
      guidance: guidance.value,
      timestamp: new Date().toISOString(),
    }

    src.value = newImage.src
    generationHistory.value.unshift(newImage)

    toast.add({
      title: 'Success!',
      description: 'Image generated successfully',
      color: 'green',
    })
  }
  catch (error) {
    toast.add({
      title: 'Generation Failed',
      description: error.message || 'Something went wrong',
      color: 'red',
    })
  }
  finally {
    loading.value = false
  }
}

const qualityLabel = computed(() => {
  if (steps.value <= 10)
    return 'Fast'
  if (steps.value <= 20)
    return 'Balanced'
  return 'High Quality'
})

// Image upload handling
function handleImageUpload(event) {
  const file = event.target.files[0]
  if (!file)
    return

  // Validate file type
  if (!file.type.startsWith('image/')) {
    toast.add({
      title: 'Invalid File',
      description: 'Please select an image file',
      color: 'red',
    })
    return
  }

  uploadedImage.value = file
  uploadedImageUrl.value = URL.createObjectURL(file)

  // Reset mask when new image is uploaded
  maskData.value = null

  toast.add({
    title: 'Image Uploaded',
    description: 'Image ready for processing',
    color: 'green',
  })
}

// Remove uploaded image
function removeUploadedImage() {
  if (uploadedImageUrl.value) {
    URL.revokeObjectURL(uploadedImageUrl.value)
  }
  uploadedImage.value = null
  uploadedImageUrl.value = ''
  maskData.value = null
}

// Open mask editor
function openMaskEditor() {
  if (!uploadedImageUrl.value) {
    toast.add({
      title: 'No Image',
      description: 'Please upload an image first',
      color: 'red',
    })
    return
  }
  showMaskEditor.value = true
}

// Handle mask data from editor
function handleMaskData(mask) {
  maskData.value = mask
  showMaskEditor.value = false
  toast.add({
    title: 'Mask Applied',
    description: 'Mask is ready for inpainting',
    color: 'green',
  })
}

// Convert file to base64
function fileToBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => resolve(reader.result.split(',')[1]) // Remove data:image/...;base64, prefix
    reader.onerror = reject
    reader.readAsDataURL(file)
  })
}
</script>

<template>
  <div class="min-h-screen py-12">
    <div class="mx-auto max-w-6xl px-4">
      <!-- Header -->
      <div class="text-center mb-12">
        <div class="flex items-center justify-center mb-6">
          <div class="relative">
            <div
              class="w-16 h-16 rounded-full bg-gradient-to-br from-green-500/20 to-teal-500/20 flex items-center justify-center">
              <UIcon name="i-lucide-settings" class="size-8 text-green-400" />
            </div>
            <div class="absolute -top-1 -right-1 animate-pulse">
              <div
                class="w-6 h-6 bg-gradient-to-r from-green-400 to-teal-400 rounded-full flex items-center justify-center">
                <UIcon name="i-lucide-sparkles" class="size-3 text-black animate-spin" />
              </div>
            </div>
          </div>
        </div>

        <h1
          class="text-4xl sm:text-5xl font-bold bg-gradient-to-r from-green-400 via-teal-400 to-blue-400 bg-clip-text text-transparent mb-4">
          NSFW Clothes Remover
        </h1>

        <p class="text-lg text-muted max-w-2xl mx-auto mb-6">
          State-of-the-art diffusion-based clothes removal
        </p>

        <div class="flex items-center justify-center gap-6 text-sm">
          <div class="flex items-center gap-2 text-green-400">
            <UIcon name="i-lucide-settings" class="size-4" />
            <span>Customizable</span>
          </div>
          <div class="flex items-center gap-2 text-blue-400">
            <UIcon name="i-lucide-image" class="size-4" />
            <span>High Resolution</span>
          </div>
          <div class="flex items-center gap-2 text-purple-400">
            <UIcon name="i-lucide-sliders" class="size-4" />
            <span>Advanced Controls</span>
          </div>
        </div>
      </div>

      <div class="space-y-6">
        <!-- Input Form -->
        <SpotlightCard>
          <div class="flex items-center justify-between mb-6">
            <h3 class="text-xl font-semibold text-white flex items-center gap-2">
              <UIcon name="i-lucide-edit" class="size-5 text-blue-400" />
              Create Your Image
            </h3>
            <div class="flex items-center gap-2">
              <div class="w-2 h-2 rounded-full bg-green-500 animate-pulse"></div>
              <span class="text-xs text-green-400 font-medium">Ready</span>
            </div>
          </div>

          <form class="space-y-6" @submit.prevent="generateImage">
            <!-- Image Upload -->
            <div>
              <label class="block text-sm font-medium text-white mb-2">
                Upload Image
                <span class="text-red-400">*</span>
              </label>

              <!-- Upload Area -->
              <div v-if="!uploadedImageUrl"
                class="relative border-2 border-dashed border-gray-600 rounded-lg p-6 text-center hover:border-gray-500 transition-colors">
                <input type="file" accept="image/*" class="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                  @change="handleImageUpload" />
                <UIcon name="i-lucide-upload" class="size-8 text-gray-400 mx-auto mb-2" />
                <p class="text-white font-medium mb-1">
                  Click to upload image
                </p>
                <p class="text-xs text-muted">
                  PNG, JPG, JPEG, WEBP
                </p>
              </div>

              <!-- Uploaded Image Preview -->
              <div v-else class="space-y-3">
                <div class="relative inline-block">
                  <img :src="uploadedImageUrl" alt="Uploaded image"
                    class="max-w-full max-h-48 rounded-lg border border-gray-600" />
                  <button type="button"
                    class="absolute -top-2 -right-2 w-6 h-6 bg-red-500 hover:bg-red-600 rounded-full flex items-center justify-center transition-colors"
                    @click="removeUploadedImage">
                    <UIcon name="i-lucide-x" class="size-3 text-white" />
                  </button>
                </div>

                <!-- Mask Editor Button (for inpainting) -->
                <div class="flex items-center gap-3">
                  <p class="text-sm text-muted">
                    Optionally, create a mask for inpainting (forces
                    the model to only modify the masked area)
                  </p>
                  <UButton color="neutral" variant="outline" size="sm" icon="i-lucide-edit" @click="openMaskEditor">
                    {{ maskData ? 'Edit Mask' : 'Create Mask' }}
                  </UButton>
                  <span v-if="maskData" class="text-xs text-green-400 flex items-center gap-1">
                    <UIcon name="i-lucide-check" class="size-3" />
                    Mask ready
                  </span>
                  <span v-else class="text-xs text-muted">
                    Click to create inpainting mask
                  </span>
                </div>
              </div>
            </div>

            <!-- Strength Control -->
            <div>
              <div class="flex items-center justify-between mb-3">
                <label class="block text-sm font-medium text-white">Transformation Strength</label>
                <span class="text-sm text-muted">{{ strength.toFixed(2) }}</span>
              </div>
              <USlider v-model="strength" :min="0.1" :max="1.0" :step="0.05" class="w-full" />
              <div class="flex justify-between text-xs text-muted mt-1">
                <span>Keep Original</span>
                <span>Transform More</span>
              </div>
              <p class="text-xs text-muted mt-1">
                Lower values preserve more of the original image
              </p>
            </div>

            <!-- Steps Control -->
            <div>
              <div class="flex items-center justify-between mb-3">
                <label class="block text-sm font-medium text-white">Diffusion Steps</label>
                <div class="flex items-center gap-2">
                  <span class="text-sm text-muted">{{ steps }}</span>
                  <div class="px-2 py-1 bg-yellow-500/20 rounded text-xs text-yellow-400">
                    {{ qualityLabel }}
                  </div>
                </div>
              </div>
              <USlider v-model="steps" :min="1" :max="100" :step="1" class="w-full" />
              <div class="flex justify-between text-xs text-muted mt-1">
                <span>Faster</span>
                <span>Higher Quality</span>
              </div>
            </div>

            <!-- Advanced Settings -->
            <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label class="block text-sm font-medium text-white mb-2">Guidance Scale</label>
                <UInput v-model.number="guidance" type="number" :min="1" :max="10" :step="0.1" class="w-full" />
                <p class="text-xs text-muted mt-1">
                  Higher values follow the prompt (clothes removal)
                  more closely (risk of
                  degraded quality for high
                  values)
                </p>
              </div>
              <div>
                <label class="block text-sm font-medium text-white mb-2">Seed (optional)</label>
                <div class="flex gap-2">
                  <UInput v-model="seed" type="number" placeholder="Random" class="flex-1" />
                </div>
                <p class="text-xs text-muted mt-1">
                  Leave empty for random generation
                </p>
              </div>
            </div>

            <!-- Generate Button -->
            <SpotlightButton type="submit" :loading="loading" :animate="false" class="w-full py-3">
              <div class="flex items-center justify-center gap-2 relative z-10">
                <UIcon name="i-lucide-sparkles" class="size-4" :class="[loading ? 'animate-spin' : '']" />
                <span class="text-white font-medium">
                  {{
                    loading ? 'Generating...' :
                      `Generate Image`
                  }}
                </span>
              </div>
            </SpotlightButton>
          </form>
        </SpotlightCard>

        <!-- Generated Image Display -->
        <SpotlightCard v-if="generatedImages || loading">
          <div class="flex items-center justify-between mb-4">
            <h3 class="text-lg font-semibold text-white">
              Results
            </h3>
            <div v-if="!loading && generatedImages" class="flex items-center gap-2">
              <button class="p-2 hover:bg-white/10 rounded-lg transition-colors" title="Download composite"
                @click="downloadImage({ url: generatedImages.composite_image })">
                <UIcon name="i-lucide-download" class="size-4 text-gray-400" />
              </button>
              <button class="p-2 hover:bg-white/10 rounded-lg transition-colors" title="Download generated only"
                @click="downloadImage({ url: generatedImages.gen_image })">
                <UIcon name="i-lucide-image" class="size-4 text-gray-400" />
              </button>
            </div>
          </div>

          <div class="space-y-4">
            <!-- Loading State -->
            <div v-if="loading" class="aspect-square flex items-center justify-center bg-gray-900 rounded-lg">
              <div class="text-center">
                <div
                  class="w-12 h-12 border-4 border-green-400/20 border-t-green-400 rounded-full animate-spin mx-auto mb-4">
                </div>
                <p class="text-white/80">
                  Creating your masterpiece...
                </p>
                <p class="text-sm text-muted">
                  This may take a few moments
                </p>
              </div>
            </div>

            <!-- Before/After Comparison -->
            <div v-else-if="generatedImages">
              <div class="mb-3">
                <h4 class="text-sm font-medium text-white mb-1">
                  Before & After Comparison
                </h4>
                <p class="text-xs text-muted">
                  Drag the divider to compare original and generated images
                </p>
              </div>
              <BeforeAfterComparison :before-image="generatedImages.input_image"
                :after-image="generatedImages.composite_image" before-label="Original" after-label="Generated"
                :initial-position="50" />

              <!-- Debug: Individual Images -->
              <div class="mt-6 space-y-4">
                <h4 class="text-sm font-medium text-white">
                  Debug: Individual Images
                </h4>
                <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div>
                    <p class="text-xs text-gray-400 mb-2">
                      Input Image
                    </p>
                    <img :src="generatedImages.input_image" class="w-full h-auto rounded border" />
                  </div>
                  <div>
                    <p class="text-xs text-gray-400 mb-2">
                      Generated Only
                    </p>
                    <img :src="generatedImages.gen_image" class="w-full h-auto rounded border" />
                  </div>
                  <div>
                    <p class="text-xs text-gray-400 mb-2">
                      Composite
                    </p>
                    <img :src="generatedImages.composite_image" class="w-full h-auto rounded border" />
                  </div>
                  <div>
                    <p class="text-xs text-gray-400 mb-2">
                      Mask
                    </p>
                    <img :src="generatedImages.mask" class="w-full h-auto rounded border" />
                  </div>
                </div>
              </div>
            </div>
          </div>
        </SpotlightCard>

        <!-- System Metrics and Live Logs -->
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <SpotlightCard>
            <SystemMetrics />
          </SpotlightCard>
          <SpotlightCard>
            <LogStream />
          </SpotlightCard>
        </div>
      </div>
    </div>

    <!-- Mask Editor Modal -->
    <MaskEditor :open="showMaskEditor" :image-url="uploadedImageUrl" :existing-mask="maskData"
      @close="showMaskEditor = false" @save-mask="handleMaskData" />
  </div>
</template>
