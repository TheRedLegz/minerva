const pp = require('puppeteer');
const LanguageDetect = require('languagedetect');
const chalk = require('chalk');
const log = console.log;

const lngDetector = new LanguageDetect();

const getQueryString = ({
  from,
  to,
  query,
  hashtags,
  location,
  language = 'en',
}) => {
  let date = '';

  if (from && to) {
    date = `since:${from} until:${to}`;
  }

  let hashes = '';

  if (hashtags && hashtags.length > 0 && typeof hashtags == 'array') {
    hashes = '(' + hashtags.map((item) => '#' + item).join(' OR ') + ')';
  }

  let loc = '';

  if (typeof location == 'object') {
    const { lat, long, radius } = location;
    loc = `geocode:${lat},${long},${radius}km`;
  } else if (typeof location == 'string') loc = `near:${location}`;

  const lang = `lang:${language}`;

  const res = `${query} ${date} ${hashes} ${loc} ${lang}`;

  return encodeURIComponent(res);
};

const scrape = async ({ test, ...options }) => {
  const to_save = [];
  const tweets_found = [];

  const to_search = getQueryString(options);

  const browser = await pp.launch();
  const page = await browser.newPage();

  page.on('response', async (e) => {
    const url = e.url();

    if (url.includes('adaptive.json')) {
      const json = await e.json();

      const keys = Object.keys(json.globalObjects.tweets);

      log(chalk.blue('TWEETS ') + `${keys.length} received`);

      for (const i of keys) {
        const obj = json.globalObjects.tweets[i];

        const { full_text, id, lang } = obj;
        const [[lng, val]] = lngDetector.detect(full_text, 1);

        tweets_found.push(obj);

        if (lng === 'english' && val > 0.2 && lang == 'en') {
          to_save.push(obj);
        }
      }
    }
  });

  await page.setUserAgent(
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36 Edg/91.0.864.54'
  );

  const url = `https://twitter.com/search?q=${to_search}\n`;

  log(chalk.bgBlue.white('URL') + ` ${url}`);

  await page.goto(url, {
    waitUntil: 'networkidle0',
  });

  let lastHeight = await page.evaluate(() => {
    return document.body.scrollHeight;
  });

  log(chalk.bgGreen.white('\nSCRAPE') + ' started');

  while (true) {
    log(chalk.green('\nSCROLL ') + 'simulating');
    try {
      await page.evaluate((y) => {
        window.scrollTo(0, y);
      }, lastHeight);

      await page.waitForTimeout(3000);

      await page.evaluate((y) => {
        window.scrollTo(0, y);
      }, lastHeight + 50);

      const currentHeight = await page.evaluate(() => {
        return document.body.scrollHeight;
      });

      if (lastHeight == currentHeight) break;

      await page.waitForTimeout(3000);

      lastHeight = currentHeight;
      log(chalk.green('SCROLL ') + 'done');
    } catch (e) {
      console.log(e);
      break;
    }
  }

  

  await browser.close();

  log(chalk.bgGreen.white('\nSCRAPE') + ' done\n');
  log(chalk.blue('TWEET ') + tweets_found.length + ' found');
  log(chalk.blue('TWEET ') + to_save.length + ' filtered');


  return {
    tweets: tweets_found,
    filtered: to_save,
  };
};

module.exports = scrape;
