
const pp = require("puppeteer");
const fs = require("fs");
const mongoose = require('mongoose');
const RawTweet = require('./models/RawTweet')
const LanguageDetect = require('languagedetect');

const lngDetector = new LanguageDetect()

mongoose.connect('mongodb://localhost:27017/minerva_raw_tweets', {useNewUrlParser: true, useUnifiedTopology: true});


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

  let count = 0;
  const to_save = [];

  const to_search = getQueryString(options)

  const browser = await pp.launch();
  const page = await browser.newPage();

  page.on("response", async (e) => {
    const url = e.url();

    if (url.includes("adaptive.json")) {
      const json = await e.json();

      const keys = Object.keys(json.globalObjects.tweets);

      for (const i of keys) {
        count++;
        const obj = json.globalObjects.tweets[i];
        const { full_text, id, lang } = obj
        const [ [lng, val] ] = lngDetector.detect(full_text, 1)


        if(lng === 'english' && val > .2 && lang == 'en') {
          to_save.push(json.globalObjects.tweets[i]);
        }

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

      await page.waitForTimeout(4000);

      await page.evaluate((y) => {
        window.scrollTo(0, y);
      }, lastHeight + 50);


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

  console.log(`Scraping done\nNumber of tweets found: ${count}\nNumber of tweets saved: ${to_save.length}\n`)

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
  from: '2020-04-01',
  to: '2021-08-01',
  query: `"online class" OR "e-class"`,
  location: 'Cebu',
  lang: 'en'
})
