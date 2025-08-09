<script setup lang="ts">
useHead({
  title: 'AI Video Generator',
  meta: [
    { name: 'description', content: 'Generate videos from text prompts using AI' },
  ],
})

const toast = useToast()

const prompt = ref('')
const negativePrompt = ref('Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards')
const steps = ref(30)
const cfg = ref(5.0)
const seed = ref()
const loading = ref(false)
const generationHistory = ref([])
const generatedVideo = ref(null)
const config = useRuntimeConfig()

const uploadedImage = ref(null)
const uploadedImageUrl = ref('')
const maxDimension = ref(1024)
const numFrames = ref(121)
const width = ref(768)
const height = ref(768)

// Download video function
function downloadVideo() {
  if (!generatedVideo.value) return

  const link = document.createElement('a')
  link.href = generatedVideo.value
  link.download = `generated-video-${Date.now()}.mp4`
  link.click()
}

async function generateVideo() {
  if (loading.value)
    return

  // Validate requirements
  if (!prompt.value.trim()) {
    toast.add({
      title: 'Prompt Required',
      description: 'Please enter a text prompt',
      color: 'red',
    })
    return
  }

  loading.value = true

  try {
    const requestBody: any = {
      prompt: prompt.value,
      negative_prompt: negativePrompt.value,
      steps: steps.value,
      cfg: cfg.value,
      seed: seed.value ? Number.parseInt(seed.value) : -1,
      max_dimension: maxDimension.value,
      num_frames: numFrames.value,
      width: width.value,
      height: height.value,
    }

    // Add input image if provided
    if (uploadedImage.value) {
      const inputImageB64 = await fileToBase64(uploadedImage.value)
      requestBody.input_image = inputImageB64
    }

    const response = await fetch(`${config.public.apiBase}/api/video/generate`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(requestBody)
    })

    if (!response.ok) {
      const errorData = await response.json()
      throw new Error(errorData.detail || 'Video generation failed')
    }

    // Get video as blob and create URL
    const videoBlob = await response.blob()
    const videoUrl = URL.createObjectURL(videoBlob)

    generatedVideo.value = videoUrl

    const newVideo = {
      src: videoUrl,
      prompt: prompt.value,
      steps: steps.value,
      cfg: cfg.value,
      numFrames: numFrames.value,
      timestamp: new Date().toISOString(),
    }

    generationHistory.value.unshift(newVideo)

    toast.add({
      title: 'Success!',
      description: 'Video generated successfully',
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

// Image upload handling (for image-to-video)
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

  toast.add({
    title: 'Image Uploaded',
    description: 'Image ready for video generation',
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

// Cleanup video URLs on unmount
onUnmounted(() => {
  if (generatedVideo.value) {
    URL.revokeObjectURL(generatedVideo.value)
  }
  generationHistory.value.forEach(video => {
    if (video.src) {
      URL.revokeObjectURL(video.src)
    }
  })
})
</script>

<template>
  <div class="min-h-screen py-12">
    <div class="mx-auto max-w-6xl px-4">
      <!-- Header -->
      <div class="text-center mb-12">
        <div class="flex items-center justify-center mb-6">
          <div class="relative">
            <div
              class="w-16 h-16 rounded-full bg-gradient-to-br from-purple-500/20 to-pink-500/20 flex items-center justify-center">
              <UIcon name="i-lucide-video" class="size-8 text-purple-400" />
            </div>
            <div class="absolute -top-1 -right-1 animate-pulse">
              <div
                class="w-6 h-6 bg-gradient-to-r from-purple-400 to-pink-400 rounded-full flex items-center justify-center">
                <UIcon name="i-lucide-sparkles" class="size-3 text-black animate-spin" />
              </div>
            </div>
          </div>
        </div>

        <h1
          class="text-4xl sm:text-5xl font-bold bg-gradient-to-r from-purple-400 via-pink-400 to-red-400 bg-clip-text text-transparent mb-4">
          AI Video Generator
        </h1>

        <p class="text-lg text-muted max-w-2xl mx-auto mb-6">
          Generate high-quality videos from text prompts using WAN 2.2 Video 5B model
        </p>

        <div class="flex items-center justify-center gap-6 text-sm">
          <div class="flex items-center gap-2 text-purple-400">
            <UIcon name="i-lucide-wand-2" class="size-4" />
            <span>Text-to-Video</span>
          </div>
          <div class="flex items-center gap-2 text-pink-400">
            <UIcon name="i-lucide-image" class="size-4" />
            <span>Image-to-Video</span>
          </div>
          <div class="flex items-center gap-2 text-red-400">
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
              <UIcon name="i-lucide-edit" class="size-5 text-purple-400" />
              Create Your Video
            </h3>
            <div class="flex items-center gap-2">
              <div class="w-2 h-2 rounded-full bg-purple-500 animate-pulse"></div>
              <span class="text-xs text-purple-400 font-medium">Ready</span>
            </div>
          </div>

          <form class="space-y-6" @submit.prevent="generateVideo">
            <!-- Prompt Input -->
            <div>
              <label class="block text-sm font-medium text-white mb-2">
                Prompt
                <span class="text-red-400">*</span>
              </label>
              <UTextarea v-model="prompt"
                placeholder="A majestic eagle soaring through mountain peaks, cinematic shot, 4k quality..." rows="3"
                class="w-full" />
              <p class="text-xs text-muted mt-1">
                Describe the video you want to generate in detail
              </p>
            </div>

            <!-- Negative Prompt -->
            <div>
              <label class="block text-sm font-medium text-white mb-2">
                Negative Prompt
              </label>
              <UTextarea v-model="negativePrompt" placeholder="Things to avoid in the video..." rows="2"
                class="w-full" />
              <p class="text-xs text-muted mt-1">
                Describe what you don't want to see in the video
              </p>
            </div>

            <!-- Optional Image Upload for Image-to-Video -->
            <div>
              <label class="block text-sm font-medium text-white mb-2">
                Input Image (optional)
              </label>
              <p class="text-xs text-muted mb-3">
                Upload an image to use as the first frame for image-to-video generation
              </p>

              <!-- Upload Area -->
              <div v-if="!uploadedImageUrl"
                class="relative border-2 border-dashed border-gray-600 rounded-lg p-6 text-center hover:border-gray-500 transition-colors">
                <input type="file" accept="image/*" class="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                  @change="handleImageUpload" />
                <UIcon name="i-lucide-upload" class="size-8 text-gray-400 mx-auto mb-2" />
                <p class="text-white font-medium mb-1">
                  Click to upload first frame (optional)
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
                <p class="text-sm text-green-400">
                  First frame uploaded - will generate image-to-video
                </p>
              </div>
            </div>

            <!-- Video Settings -->
            <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              <div>
                <label class="block text-sm font-medium text-white mb-2">Width</label>
                <UInput v-model.number="width" type="number" :min="64" :max="2048" :step="8" class="w-full" />
              </div>
              <div>
                <label class="block text-sm font-medium text-white mb-2">Height</label>
                <UInput v-model.number="height" type="number" :min="64" :max="2048" :step="8" class="w-full" />
              </div>
              <div>
                <label class="block text-sm font-medium text-white mb-2">Max Dimension</label>
                <UInput v-model.number="maxDimension" type="number" :min="256" :max="2048" :step="32" class="w-full" />
              </div>
            </div>

            <!-- Frame Count -->
            <div>
              <div class="flex items-center justify-between mb-3">
                <label class="block text-sm font-medium text-white">Number of Frames</label>
                <span class="text-sm text-muted">{{ numFrames }} frames</span>
              </div>
              <USlider v-model="numFrames" :min="1" :max="121" :step="1" class="w-full" />
              <div class="flex justify-between text-xs text-muted mt-1">
                <span>Short</span>
                <span>Long</span>
              </div>
              <p class="text-xs text-muted mt-1">
                More frames = longer video (at 24fps: {{ Math.round(numFrames / 24 * 10) / 10 }}s)
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
                <label class="block text-sm font-medium text-white mb-2">CFG Scale</label>
                <UInput v-model.number="cfg" type="number" :min="1" :max="20" :step="0.1" class="w-full" />
                <p class="text-xs text-muted mt-1">
                  Higher values follow the prompt more closely
                </p>
              </div>
              <div>
                <label class="block text-sm font-medium text-white mb-2">Seed (optional)</label>
                <UInput v-model="seed" type="number" placeholder="Random" class="w-full" />
                <p class="text-xs text-muted mt-1">
                  Leave empty for random generation
                </p>
              </div>
            </div>

            <!-- Generate Button -->
            <SpotlightButton type="submit" :loading="loading" :animate="false" class="w-full py-3">
              <div class="flex items-center justify-center gap-2 relative z-10">
                <UIcon name="i-lucide-video" class="size-4" :class="[loading ? 'animate-pulse' : '']" />
                <span class="text-white font-medium">
                  {{ loading ? 'Generating Video...' : 'Generate Video' }}
                </span>
              </div>
            </SpotlightButton>
          </form>
        </SpotlightCard>

        <!-- Generated Video Display -->
        <SpotlightCard v-if="generatedVideo || loading">
          <div class="flex items-center justify-between mb-4">
            <h3 class="text-lg font-semibold text-white">
              Generated Video
            </h3>
            <div v-if="!loading && generatedVideo" class="flex items-center gap-2">
              <button class="p-2 hover:bg-white/10 rounded-lg transition-colors" title="Download video"
                @click="downloadVideo">
                <UIcon name="i-lucide-download" class="size-4 text-gray-400" />
              </button>
            </div>
          </div>

          <div class="space-y-4">
            <!-- Loading State -->
            <div v-if="loading" class="aspect-video flex items-center justify-center bg-gray-900 rounded-lg">
              <div class="text-center">
                <div
                  class="w-12 h-12 border-4 border-purple-400/20 border-t-purple-400 rounded-full animate-spin mx-auto mb-4">
                </div>
                <p class="text-white/80">
                  Generating your video...
                </p>
                <p class="text-sm text-muted">
                  This may take several minutes
                </p>
              </div>
            </div>

            <!-- Video Player -->
            <div v-else-if="generatedVideo" class="relative">
              <video :src="generatedVideo" controls class="w-full rounded-lg border border-gray-600" preload="metadata">
                Your browser does not support the video tag.
              </video>
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
  </div>
</template>