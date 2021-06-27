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

// save as json for checking
// TODO connect to db

// save everything in the db to the database 

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
  } else loc = `near:${location}`
  
  const lang = `lang:${language}`

  const res = `${query} ${date} ${hashes} ${loc} ${lang}`


  return encodeURIComponent(res)
}

const scrape = async (options) => {

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

  await page.goto(
    `https://twitter.com/search?q=${to_search}`,
    {
      waitUntil: "networkidle0",
    }
  );

  let lastHeight = await page.evaluate(() => {
    return document.body.scrollHeight;
  });

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

  to_save.forEach(async (item, i) => {
    console.log('Saving Tweet ' + i)
    const row = new RawTweet({ data: item })
    await row.save()
  })

  console.log('Done')
};


scrape({
  from: '2021-05-01',
  to: '2021-06-01',
  query: `"online classes" OR "e-class"`,
  location: {
    lat: 10.3095549,
    long: 123.8931107,
    radius: 5
  }
})

