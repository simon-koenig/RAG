<script lang="ts">
    import * as Select from "$lib/components/ui/select";
 
   
  let {
      questions
    } : {
      questions: string[];
     
    } =  $props();
</script>


<article>
    
  <Select.Root>
    <Select.Trigger class="w-[180px]">
      <Select.Value placeholder="Example Questions" />
    </Select.Trigger>
    <Select.Content>
      {#each questions as question}
      <Select.Item value={question}>{question}</Select.Item>
      {/each}
    </Select.Content>
  </Select.Root>
  


</article>