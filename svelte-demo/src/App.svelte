<script>
  import { onMount } from 'svelte';

  let messages = [];
  let query = '';
  let OPENAIMODEL = "mixtral";
  let ENDPOINT = "http://10.103.251.104:8040/v1";
  let MARQO_ENDPOINT = "http://10.103.251.100:8882";
  let APIKEY = "N/A";
  let INDEX = "animal-facts"

  const addMessage = (role, content) => {
    messages = [...messages, { role, content }];
  };

  const sendMessage = async () => {
    addMessage('user', query);

    const data = {
      model: OPENAIMODEL,
      messages: [{ role: 'user', content: query }]
    };
    console.log(data);
    try {
      const response = await fetch(ENDPOINT + '/chat/completions', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          // 'Authorization': "Bearer N/A"
        },
        body: JSON.stringify(data)
      });

      const report = await response.json();

      if (report.choices.length > 0) {
        const result = report.choices[0].message.content;
        addMessage('assistant', result);
      } else {
        addMessage('assistant', 'No result generated!');
      }
    } catch (error) {
      console.error('Error:', error);
      addMessage("assistant", error);
      // addMessage('assistant', 'Error: Failed to get response from server');
    }
  };


  const mqLookUp = async () => {
    const searchEndpoint = MARQO_ENDPOINT + "/indexes/" + INDEX + "/search";
    const searchData = {
      q: query,
      searchableAttributes: ["Text"],
      searchMethod: 'TENSOR'
    };
    try {
      const response = await fetch(searchEndpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(searchData)
      });

      if (!response.ok) {
        throw new Error('Network response was not ok');
      }

      const data = await response.json();
      console.log(data); // Handle the response data here
    } catch (error) {
      console.error('Error fetching data:', error);
    }
  }

function handleSearchResults() {
  let numhits = results.hits.length;
  let durldict = {};
  let background = "";
  let sources = "";
  let numtokens = answer_size + (query.length / 4);
  const MAXTOKENS = /* specify your maximum tokens value here */;
  const num_ref = /* specify your num_ref value here */;
  const query_threshold = /* specify your query_threshold value here */;

  if (numhits > 0) {
    let num_sources = 0;
    for (let i = 0; i < numhits && i < num_ref; i++) {
      let score = parseFloat(results.hits[i]._score);
      if (score >= query_threshold) {
        let fuid = results.hits[i].fuid;
        let durl = durldict[fuid];
        if (!durl) {
          // Acces mino client
          // durl = minioclient.get_presigned_url(
          //               "GET",
          //               index,
          //               fuid,
          //               expires=datetime.timedelta(days=1),
          //               response_headers={"response-content-type": "application/pdf"},
          //           )
          //           durldict[fuid] = durl

          // You need to handle the logic for getting the presigned URL here
          // Example: 
          // durl = await getPresignedUrl(fuid);
          // function getPresignedUrl(fuid) {
          //    // Your logic to get the presigned URL
          // }
          // durldict[fuid] = durl;
        }
        numtokens += results.hits[i].tokens;
        let sourcetext = escapeMarkdown(results.hits[i].Text);
        if (numtokens < MAXTOKENS) {
          let scorestring = score.toFixed(2);
          let refstring = `[${results.hits[i].Title}, page ${results.hits[i].Page}, paragraph ${results.hits[i].Paragraph}]\n`;
          sources += `${i + 1}. ${sourcetext} [${refstring}](${durl}) (Score: ${scorestring})\n`;
          background += results.hits[i].Text + " ";
          num_sources++;
        } else {
          console.log("Model token limit exceeded, sources reduced to " + i);
          break;
        }
      }
    }
  }
}

  onMount(() => {
    // Initialize messages with initial message
    addMessage('assistant', 'Ask me Anything! Frage mich etwas!');
  });
</script>

  <svelte:head>
    <title>Simons Chatbot</title>
  </svelte:head>

<main>
  <div>
    <h1>Simple Chatbot</h1>
    <p>🕹️ Try me! But be careful, my answers might be incorrect and I can give you no warranties! 🕹️</p>

    {#each messages as message}
      {#if message.role === 'assistant'}
        <div><strong>Assistant:</strong> {message.content}</div>
      {:else if message.role === 'user'}
        <div><strong>User:</strong> {message.content}</div>
      {/if}
    {/each}

    <textarea bind:value={query} placeholder="Type your message here..."></textarea>
    <button on:click={sendMessage}>Send</button>
    <button on:click={mqLookUp}>Marqo Search</button>
  </div>
</main>





<style>
	main {
		text-align: center;
		padding: 1em;
		max-width: 240px;
		margin: 0 auto;
	}

  textarea {
    width: 400px;
  }

	h1 {
		color: #000;
		text-transform: uppercase;
		font-size: 4em;
		font-weight: 100;
	}

	@media (min-width: 640px) {
		main {
			max-width: none;
		}
	}
</style>