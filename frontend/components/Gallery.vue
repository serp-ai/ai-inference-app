<script setup lang="ts">
  interface Image {
    id: string
    url: string
    prompt: string
    style: string
    steps: number
    tokenCost?: number
    createdAt: string
    size?: number
    filename?: string
    isPublic?: boolean
  }

  interface Props {
    title?: string
    images: Image[]
    loading?: boolean
    hasMore?: boolean
    showLoadMore?: boolean
    compact?: boolean
    showTokenCost?: boolean
    emptyMessage?: string
    emptySubMessage?: string
  }

  const props = withDefaults(defineProps<Props>(), {
    title: 'Gallery',
    images: () => [],
    loading: false,
    hasMore: false,
    showLoadMore: true,
    compact: false,
    showTokenCost: true,
    emptyMessage: 'No images yet',
    emptySubMessage: 'Images will appear here once generated',
  })

  const emit = defineEmits<{
    loadMore: []
    refresh: []
  }>()

  // Modal state
  const selectedImage = ref<Image | null>(null)
  const showModal = ref(false)

  function openImageModal(image: Image) {
    selectedImage.value = image
    showModal.value = true
  }

  // Download image
  function downloadImage(image: Image) {
    const link = document.createElement('a')
    link.href = image.url
    link.download = image.filename || `generated-${Date.now()}.png`
    link.click()
  }

  // Format date
  function formatDate(dateString: string) {
    return new Date(dateString).toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    })
  }
</script>

<template>
  <SpotlightCard>
    <div class="flex items-center justify-between mb-6">
      <h3 class="text-xl font-semibold text-white flex items-center gap-2">
        <UIcon name="i-lucide-images" class="size-5 text-purple-400" />
        {{ title }}
      </h3>
      <button :disabled="loading" class="p-2 hover:bg-white/10 rounded-lg transition-colors" @click="emit('refresh')">
        <UIcon name="i-lucide-refresh-cw" class="size-4 text-gray-400" :class="[loading ? 'animate-spin' : '']" />
      </button>
    </div>

    <!-- Loading state -->
    <div v-if="loading" class="grid grid-cols-2 md:grid-cols-3 gap-4">
      <div v-for="i in 6" :key="i" class="aspect-square bg-gray-800 rounded-lg animate-pulse"></div>
    </div>

    <!-- Empty state -->
    <div v-else-if="images.length === 0" class="text-center py-8">
      <UIcon name="i-lucide-image-off" class="size-12 text-gray-600 mx-auto mb-3" />
      <p class="text-muted mb-2">
        {{ emptyMessage }}
      </p>
      <p class="text-muted text-sm">
        {{ emptySubMessage }}
      </p>
    </div>

    <!-- Images grid -->
    <div v-else>
      <div
        class="grid gap-4"
        :class="[
          compact ? 'grid-cols-3 md:grid-cols-4' : 'grid-cols-2 md:grid-cols-3 lg:grid-cols-4',
        ]"
      >
        <div
          v-for="image in images"
          :key="image.id"
          class="group relative aspect-square bg-gray-900 rounded-lg overflow-hidden hover:ring-2 hover:ring-blue-500/50 transition-all cursor-pointer"
          @click="openImageModal(image)"
        >
          <!-- Image -->
          <img
            :src="image.url"
            :alt="image.prompt"
            class="w-full h-full object-cover"
            loading="lazy"
          />

          <!-- Overlay -->
          <div
            class="absolute inset-0 bg-black/60 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center"
          >
            <div class="flex gap-2">
              <button
                class="p-2 bg-white/20 hover:bg-white/30 rounded-lg transition-colors"
                title="Download"
                @click.stop="downloadImage(image)"
              >
                <UIcon name="i-lucide-download" class="size-4 text-white" />
              </button>
            </div>
          </div>

          <!-- Info -->
          <div class="absolute bottom-0 left-0 right-0 p-2 bg-gradient-to-t from-black/80 to-transparent">
            <p class="text-white text-xs truncate mb-1">
              {{ image.prompt }}
            </p>
            <div class="flex items-center justify-between text-xs text-gray-300">
              <span>{{ formatDate(image.createdAt) }}</span>
              <span v-if="showTokenCost && image.tokenCost">{{ image.tokenCost }} tokens</span>
            </div>
          </div>
        </div>
      </div>

      <!-- Load more button -->
      <div v-if="hasMore && showLoadMore" class="text-center mt-6">
        <SpotlightButton
          :disabled="loading"
          :animate="false"
          transparent
          class="px-6 py-2 border border-white/20"
          @click="emit('loadMore')"
        >
          <div class="flex items-center gap-2 relative z-10">
            <UIcon name="i-lucide-plus" class="size-4" :class="[loading ? 'animate-spin' : '']" />
            <span class="text-white">{{ loading ? 'Loading...' : 'Load More' }}</span>
          </div>
        </SpotlightButton>
      </div>
    </div>
  </SpotlightCard>

  <!-- Image Modal -->
  <UModal v-model:open="showModal" :ui="{ content: 'max-w-4xl' }">
    <template v-if="selectedImage" #content="{ close }">
      <div class="p-6 space-y-4">
        <!-- Image -->
        <div class="relative">
          <img
            :src="selectedImage.url"
            :alt="selectedImage.prompt"
            class="w-full h-auto max-h-[60vh] object-contain rounded-lg"
          />
        </div>

        <!-- Image Details -->
        <div class="space-y-4">
          <div>
            <h3 class="text-lg font-semibold text-white mb-2">
              Generation Details
            </h3>
            <p class="text-gray-300 text-sm leading-relaxed">
              {{ selectedImage.prompt }}
            </p>
          </div>

          <div class="grid grid-cols-2 md:grid-cols-4 gap-4 pt-4 border-t border-gray-700">
            <div class="text-center">
              <div class="text-sm text-gray-400 mb-1">
                Style
              </div>
              <div class="text-white font-medium capitalize">
                {{ selectedImage.style || 'None' }}
              </div>
            </div>
            <div class="text-center">
              <div class="text-sm text-gray-400 mb-1">
                Steps
              </div>
              <div class="text-white font-medium">
                {{ selectedImage.steps }}
              </div>
            </div>
            <div v-if="showTokenCost && selectedImage.tokenCost" class="text-center">
              <div class="text-sm text-gray-400 mb-1">
                Token Cost
              </div>
              <div class="text-white font-medium">
                {{ selectedImage.tokenCost }}
              </div>
            </div>
            <div class="text-center">
              <div class="text-sm text-gray-400 mb-1">
                Created
              </div>
              <div class="text-white font-medium">
                {{ formatDate(selectedImage.createdAt) }}
              </div>
            </div>
          </div>

          <!-- Actions -->
          <div class="flex justify-center gap-3 pt-4">
            <UButton
              color="primary"
              icon="i-lucide-download"
              @click="downloadImage(selectedImage)"
            >
              Download
            </UButton>
          </div>
        </div>
      </div>
    </template>
  </UModal>
</template>
