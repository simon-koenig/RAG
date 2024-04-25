<script lang="ts">
	import { goto } from '$app/navigation';
	const { data } = $props();
	const { bucketName, fileName } = data;

	let pdfUrl = $state('');
	let loading = $state(false);

	function navigateBack() {
		goto(`/minio/${bucketName}`);
	}

	$effect(() => {
		fetchObject();
	});

	const fetchObject = async () => {
		loading = true;
		try {
			const response = await fetch(
				'/api/minio?bucketName=' + bucketName + '&fileName=' + fileName,
				{
					method: 'GET'
				}
			);
			if (!response.ok) {
				throw new Error('Failed to fetch object');
			}
			const arrayBuffer = await response.arrayBuffer();
			const blob = new Blob([arrayBuffer], { type: 'application/pdf' });
			pdfUrl = URL.createObjectURL(blob);
		} catch (error) {
			console.error('Failed to fetch object:', error);
		} finally {
			loading = false;
		}
	};
</script>

<button class="text-4xl -translate-y-4" onclick={navigateBack} style="cursor: pointer;"
	>&#8592</button
>

{#if pdfUrl}
	<iframe title="Document Preview" src={pdfUrl} class="w-full h-[800px] rounded-lg"></iframe>
{:else if loading}
	<div class="w-full h-full flex items-center justify-center">
		<div class="spinner"></div>
	</div>
{:else}
	<div class="w-full h-full flex items-center justify-center">
		<p class="">Failed to fetch object.</p>
	</div>
{/if}

<style>
	.spinner {
		@apply w-12 h-12 border-4 border-solid border-gray-400 border-t-4 border-t-transparent rounded-full;
		animation: spin 1s linear infinite;
	}
	@keyframes spin {
		0% {
			transform: rotate(0deg);
		}
		100% {
			transform: rotate(360deg);
		}
	}
</style>
