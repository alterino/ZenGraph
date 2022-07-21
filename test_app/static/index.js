const puppeteer = require('puppeteer');
const parseString = require('xml2js').parseString;
const util = require('util');
const jquery = require('jquery');
const DOMParser = require('dom-parser');

const username = "mmarino@carneycontracting.com";
const passwd = "Luke+23:34!?\"";
const login_url = "https://auth.construction.com/login?returnUrl=https:%2F%2Fapps.construction.com%2Flogin%3FreturnUrl%3D%2F"
const login_button = `body > app-root > ng-component > div > div.col-md-6.col-sm-12.extra_padding > 
                      div.login_form.text-left.pb-4 > form > button`
const action_stage = `app-side-filter-item:nth-child(5) > app-filter-section-wrapper
                      > div > div.filter-section-heading`
const pre_design_cb = `body > app-root > div > ng-component > app-projects-section > div > 
                       ng-component > app-side-filter > app-search-filters > div > div.filter-content > div >
                       div:nth-child(2) > div.filter-subsection > app-side-filter-item:nth-child(5) >
                       app-filter-section-wrapper > div > div.filter-section-wrapper-content >
                       app-filter-section:nth-child(1) > div > div.filter-section-content >
                       app-filter-multi-checkbox:nth-child(2) > div > label`
const design_cb = `body > app-root > div > ng-component > app-projects-section > div > ng-component >
                   app-side-filter > app-search-filters > div > div.filter-content > div > div:nth-child(2) >
                   div.filter-subsection > app-side-filter-item:nth-child(5) > app-filter-section-wrapper >
                   div > div.filter-section-wrapper-content > app-filter-section:nth-child(1) > div >
                   div.filter-section-content > app-filter-multi-checkbox:nth-child(4) > div > label`
const bid_cb = `body > app-root > div > ng-component > app-projects-section > div > ng-component >
                app-side-filter > app-search-filters > div > div.filter-content > div > div:nth-child(2) >
                div.filter-subsection > app-side-filter-item:nth-child(5) > app-filter-section-wrapper > div >
                div.filter-section-wrapper-content > app-filter-section:nth-child(1) > div >
                div.filter-section-content > app-filter-multi-checkbox:nth-child(6) > div > label`
const next_btn = `body > app-root > div > ng-component > app-projects-section > div > ng-component > 
                  div > app-subpage-view-controls > 
                  div.d-flex.flex-row.justify-content-between.align-items-end.mb-2.controls > 
                  div:nth-child(5) > app-pagination > nav > ul > li:nth-child(4) > a`

const count_selector = `body > app-root > div > ng-component > app-projects-section > div > ng-component > 
                        div > app-insights > div > div.nav-item.active > a > span.count`

const projects_btn = '#top-left-nav > a > span'
const fs = require('fs');

var projectURLs = [];

//const preparePageForTests = async (page) => {
//    const userAgent = 'Mozilla/5.0 (X11; Linux x86_64)' +
//          'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/64.0.3282.39 Safari/537.36';
//    await page.setUserAgent(userAgent);
//
//
//
//}

var logStream = fs.createWriteStream('log.txt', {flags: 'a'});

(async () => {
  const browser = await puppeteer.launch({ headless: false});
  const page = await browser.newPage();
  await page.goto(login_url);
  
  await page.type('#email', username);
  await page.type('#password', passwd);
  await Promise.all([
        await page.click(login_button),
        page.waitForNavigation({ waitUntil: 'networkidle0' }),
      ]);

  await page.goto('https://apps.construction.com/projects/all', {
    waitUntil: 'networkidle0',
  });
  
  await page.click(action_stage);
  await page.click(pre_design_cb);
  await page.click(design_cb);

  await page.setRequestInterception(true)
  
  page.on('request', (request) => {
    //console.log('>>', request.method(), request.url())
    request.continue()
  })

  page.on('response', async(response) => {
      try {
        const request = response.request();
        if (request.url().includes('cc/legacy/project/search')){
          const data = await response.text();
          let pattern = /<result-item-data key="p-dr-nbr">\s*(\d*)/;
          if ((match = pattern.exec(data)) != null) {
            extractDodgeIDs(data);
          }
        }
        if (request.url().includes('cc/legacy/project/report')){
          const data = await response.text();
          let pattern = /<dr-nbr>(\d*)/;
          if ((match = pattern.exec(data)) != null) {
            extractDodgeIDs(data);
          }
        }
      } catch(error) {
        console.log(error);
    }
  });
  await page.click(bid_cb, {
    waitUntil: 'networkidle2',
  });
  
  var endOfResults = false;
  var clickCount = 0;
  const nav_btn_sel = 'div'

  await preparePageForTests(page);
  page.screenshot({ path: "./screenshot.jpg", type: "jpeg", fullPage: true });
  //const test = 'ul.pagination > li >  a';
  //const test = `body > app-root > div > ng-component > app-projects-section > div > 
  //              ng-component > div > app-subpage-view-controls > 
  //              div.d-flex.flex-row.justify-content-between.align-items-end.mb-2.controls > 
  //              div:nth-child(5) > app-pagination > nav > ul > li:nth-child(5) > a`
  const test = `app-pagination > nav > ul > li:nth-child(5) > a`
  await page.waitForSelector(test, {
    visible: true,
  });
  const testpage2 = `app-pagination > nav > ul > li:nth-child(7) > a`
  const testpage3 = `app-pagination > nav > ul > li:nth-child(8) > a`
  const testpage4 = `app-pagination > nav > ul > li:nth-child(9) > a`
  const testpage5 = `app-pagination > nav > ul > li:nth-child(9) > a`
  const testpage6 = `app-pagination > nav > ul > li:nth-child(9) > a`
  //page.screenshot({ path: "./screenshot2.jpg", type: "jpeg", fullPage: true });
  //const testBtns = await page.evaluate(() => {
  //    const test = 'ul.pagination > a';
  //    return document.querySelector(test);
  //});
  page.screenshot({ path: "./screenshot3.jpg", type: "jpeg", fullPage: true });
  await Promise.all([
        await page.click(test),
        page.waitForNavigation({ waitUntil: 'networkidle0' }),
      ]);
  await Promise.all([
        await page.click(testpage2),
        page.waitForNavigation({ waitUntil: 'networkidle0' }),
      ]);
  await Promise.all([
        await page.click(testpage3),
        page.waitForNavigation({ waitUntil: 'networkidle0' }),
      ]);
  await Promise.all([
        await page.click(testpage4),
        page.waitForNavigation({ waitUntil: 'networkidle0' }),
      ]);
  await Promise.all([
        await page.click(testpage5),
        page.waitForNavigation({ waitUntil: 'networkidle0' }),
      ]);
  //const testBtns = await page.$(nav_btn_sel);
  //console.log('inspecting element...');
  //console.log(testBtns);
  //console.log(result);
  //testBtns.forEach(obj => {
  //  const result = Object.keys(obj).filter(key => key !== "@attributes")[0]
  //  console.log(result)
  //});
  
  //while (!endOfResults) {
  //  try {
  //      console.log(clickCount++, 'trying next button again...')
  //      await page.$(next_btn)
  //  } catch {
  //      endOfResults = true;
  //  }
  //}
})();

function extractDodgeIDs(data) {
    let pattern = /<result-item-data key="p-dr-nbr">\s*(\d*)/g;
    let match;
    let dodgeIDs = [];
    while ((match = pattern.exec(data)) != null) {
        dodgeIDs.push(match[1])
    }
    buildProjectURLs(dodgeIDs);
}

function buildProjectURLs(identifiers) {
    let baseURL = "https://apps.construction.com/projects/";

    for (let i = 0; i < identifiers.length; i++) {
        tmpURL = baseURL + identifiers[i];
        projectURLs.push(tmpURL);
        console.log(i, tmpURL)
    }
    for (let i = 0; i < projectURLs.length; i++) {
        console.log(i, projectURLs[i])
    }
}

async function getProjectData(URLs) {
    let tmpData =  [];

    for (let i = 0; i < URLs.length; i++) {
        await page.goto(URLs)
    }
}

