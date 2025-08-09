<script setup lang="ts">
  const showButton = ref(false)

  function scrollToTop() {
    window.scrollTo({
      top: 0,
      behavior: 'smooth',
    })
  }

  function handleScroll() {
    showButton.value = window.pageYOffset > 100
  }

  onMounted(() => {
    window.addEventListener('scroll', handleScroll)
  })

  onBeforeUnmount(() => {
    window.removeEventListener('scroll', handleScroll)
  })
</script>

<template>
  <div>
    <UButton v-if="showButton" class="scroll-to-top-btn bg-primary" @click="scrollToTop">
      <UIcon name="i-lucide-arrow-up" class="size-6" />
    </UButton>
  </div>
</template>

<style>
.scroll-to-top-btn {
  position: fixed;
  bottom: 20px;
  right: 20px;
  padding: 10px;
  border-radius: 50%;
  color: white;
  font-size: 16px;
  cursor: pointer;
}
</style>
