import {json} from '@sveltejs/kit'; 

// POST Route api for marqo index search

/** @type {import('./$types').RequestHandler} */
export async function POST({request}) {
  let MARQO_ENDPOINT =  "http://10.103.251.100:8882";
  let INDEX = "animal-facts";
  const searchData = await request.json();

  const searchEndpoint = `${MARQO_ENDPOINT}/indexes/${INDEX}/search`;
  console.log("Hi")
  console.log(searchData.query);
  console.log(searchEndpoint);

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
    return json(data);
  } catch (error) {
    console.error('Error fetching data:', error);
    return json(error);
  }
}
