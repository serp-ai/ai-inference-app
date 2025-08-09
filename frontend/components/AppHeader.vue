<template>
  <div class="mx-auto my-2 flex w-full items-center justify-center">
    <header class="rounded-full">
      <SpotlightButton
        rounded
        transparent
        :animate="false"
        class="border border-white/10"
      >
        <nav
          class="z-10 flex h-[50px] justify-around gap-2 p-1 transition-all duration-300 ease-in-out sm:h-[45px] sm:hover:gap-4"
        >
          <NuxtLink
            to="/"
            :class="[
              $route.path === '/'
                ? 'border border-white/5 bg-zinc-900/10 text-white/75 shadow-2xl shadow-white/50 backdrop-blur-3xl text-shadow-sm'
                : 'text-muted',
            ]"
            class="flex items-center rounded-full border border-transparent px-4 py-1 transition-all duration-300 ease-in-out hover:border-white/5 hover:bg-zinc-900/50 hover:backdrop-blur-3xl sm:px-6"
            aria-label="Home navigation link"
          >
            <UTooltip text="Home" :delay-duration="0">
              <UIcon name="i-lucide-home" class="size-7 font-light sm:size-6" />
            </UTooltip>
          </NuxtLink>
        </nav>
      </SpotlightButton>
    </header>
  </div>
</template>
