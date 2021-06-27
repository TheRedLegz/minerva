const pp = require("puppeteer");
const fs = require("fs");
const mongoose = require('mongoose');
const RawTweet = require('./models/RawTweet')

mongoose.connect('mongodb://localhost:27017/minerva_raw_tweets', {useNewUrlParser: true, useUnifiedTopology: true});


// goto the twitter search url - done
// get the adaptive json response - done
// save it - done
// check scroll position - if cant scroll, quit. if can, continue /
// scroll /
// check for adaptive - if yes, go back to step 2. if no, go back to step 4 /

// connect to db /

// save everything in the db to the database /


// TODO make a new collection for the cleaned tweets
// TODO have a more robust way of checking the language of the tweets


const to_save = [];

const getQueryString = ({ from, to, query, hashtags, location, language = 'en' }) => {

  let date = ''

  if(from && to) {
    date = `since:${from} until:${to}`
  }

  let hashes = ''

  if(hashtags && hashtags.length > 0 && typeof hashtags == 'array') {
      hashes = '(' + hashtags.map(item => '#' + item).join(' OR ') + ')'
  }
  
  let loc = ''

  if(typeof location == 'object') {
    const { lat, long, radius } = location
    loc = `geocode:${lat},${long},${radius}km`
  } else if (typeof location == 'string') loc = `near:${location}`
  
  const lang = `lang:${language}`

  const res = `${query} ${date} ${hashes} ${loc} ${lang}`


  return encodeURIComponent(res)
}

const scrape = async ({ test, ...options}) => {

  const to_search = getQueryString(options)

  const browser = await pp.launch();
  const page = await browser.newPage();

  page.on("response", async (e) => {
    const url = e.url();

    if (url.includes("adaptive.json")) {
      const json = await e.json();

      const keys = Object.keys(json.globalObjects.tweets);

      for (const i of keys) {
        to_save.push(json.globalObjects.tweets[i]);
      }
    }
  });

  await page.setUserAgent(
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36 Edg/91.0.864.54"
  );
  
  const url = `https://twitter.com/search?q=${to_search}\n`;

  console.log('Navigating to ' + url)

  await page.goto(
    url,
    {
      waitUntil: "networkidle0",
    }
  );

  let lastHeight = await page.evaluate(() => {
    return document.body.scrollHeight;
  });

  console.log('Scraping started\n')
  while (true) {
    try {
      await page.evaluate((y) => {
        window.scrollTo(0, y);
      }, lastHeight);

      await page.waitForTimeout(2000);

      const currentHeight = await page.evaluate(() => {
        return document.body.scrollHeight;
      });

      if (lastHeight == currentHeight) break;

      lastHeight = currentHeight;
    } catch (e) {
      console.log(e);
      break;
    }
  }

  await browser.close();

  console.log(`Scraping done. Number of tweets scraped: ${to_save.length}\n`)

  if(test) {
      const json = JSON.stringify(to_save, null, 2)

      console.log('Saving tweets as test_data.json\n')

      try {
        fs.writeFileSync('test/test_data.json', json)
      } catch(e) {
        console.log('Failed to write test_data.json. Error: \n')
        console.log(e)
      }

  } else {
    to_save.forEach(async (item, i) => {
      console.log('Saving Tweet ' + i + 'to DB')
      const row = new RawTweet({ data: item })
      await row.save()
    })
  }
  
  console.log('Done')
};


scrape({
  test: true,
  from: '2021-01-01',
  to: '2021-06-01',
  query: `"online class" OR "e-class"`,
})

// location: {
//   lat: 10.3095549,
//   long: 123.8931107,
//   radius: 5
// }