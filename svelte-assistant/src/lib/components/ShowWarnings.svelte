<script lang="ts">
    import * as AlertDialog from "$lib/components/ui/alert-dialog";
 
   
  let {
     warnings
    } : {
     warnings: string[];
     
    } =  $props();
</script>


<article>
    
  <AlertDialog.Root>
    <AlertDialog.Trigger>Read Me</AlertDialog.Trigger>
    <AlertDialog.Content>
      <AlertDialog.Header>
        <AlertDialog.Title>Use with Caution</AlertDialog.Title>
        <AlertDialog.Description>
          {#each warnings as warning}
            <p>{warning}</p>
          {/each}
        </AlertDialog.Description>
      </AlertDialog.Header>
      <AlertDialog.Footer>
        <AlertDialog.Cancel>Cancel</AlertDialog.Cancel>
        <AlertDialog.Action>Continue</AlertDialog.Action>
      </AlertDialog.Footer>
    </AlertDialog.Content>
  </AlertDialog.Root>

</article>