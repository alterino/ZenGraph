const fs = require('fs');
const parseString = require('xml2js').parseString;
const stripNS = require('xml2js').processors.stripPrefix;
const util = require('util');
const spawn = require("child_process").spawn;

const DOMParser = require('dom-parser');

const options = {
    tagNameProcessors: [stripNS],
    explicitArray: false,
    charkey: 'textContent',
}

const filename = 'response_sample.txt';

fs.readFile(filename, 'utf8', function(err, data) {
    if (err) throw err;
    
    extractDodgeIDs(data);
    //const pythonProcess = spawn('python',["parse_xml.py", data]);

    //pythonProcess.stdout.on('data', (data2) => {
    //    var test = new Buffer(data2, 'hex')
    //    console.log(data2);
    //    console.log(test.slice(0,8).toString());

    //});
});

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
    let projectURLs = [];

    for (let i = 0; i < identifiers.length; i++) {
        tmpURL = baseURL + identifiers[i];
        projectURLs.push(tmpURL);
    }
    
    for (let i = 0; i < projectURLs.length; i++) {
        console.log(i, projectURLs[i])
    }
}

