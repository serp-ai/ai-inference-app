<script setup lang="ts">
  defineProps({
    as: {
      type: String,
      default: 'div',
    },
    animate: {
      type: Boolean,
      default: true,
    },
    transparent: {
      type: Boolean,
      default: false,
    },
  })
</script>

<template>
  <component
    :is="as"
    class="group relative rounded-xl transition-all duration-300 p-1"
  >
    <!-- Animated border effect -->
    <div
      v-if="animate"
      class="absolute inset-0 opacity-0 transition-opacity duration-300 group-hover:opacity-100"
    >
      <div
        class="absolute inset-0 rounded-xl animate-pulse"
        style="background: conic-gradient(from 0deg at 50% 50%, transparent 0deg, rgba(255,255,255,0.1) 90deg, transparent 180deg, rgba(255,255,255,0.1) 270deg, transparent 360deg)"
      ></div>
    </div>

    <!-- Inner content with backdrop blur -->
    <div
      class="relative rounded-lg border backdrop-blur-xl transition-all duration-300 group-hover:border-white/20"
      :class="[
        transparent ? 'border-white/10' : 'border-white/10 bg-zinc-900/80',
      ]"
    >
      <!-- Subtle glow effect -->
      <div
        class="absolute inset-0 rounded-lg bg-gradient-to-br from-white/[0.02] via-transparent to-white/[0.02] opacity-0 transition-opacity duration-300 group-hover:opacity-100"
      ></div>

      <!-- Content wrapper with proper padding -->
      <div class="relative z-10 p-4">
        <slot></slot>
      </div>
    </div>
  </component>
</template>
