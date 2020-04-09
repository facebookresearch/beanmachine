"use strict";

var vows = require('vows');
var assert = require('assert');
var Remarkable = require('remarkable');

var md = new Remarkable();
var plugin = require('./index.js');

md.use(plugin);

vows.describe('KatexPlugin').addBatch({
    'Render plain text': {
        topic: md.render('This is a test.'),
        'Nothing done': function(topic) {
            assert.equal(topic, '<p>This is a test.</p>\n');
        }
    },
    'Render with single $ in text': {
        topic: md.render('The car cost $20,000 new.'),
        'Nothing done': function(topic) {
            assert.equal(topic, '<p>The car cost $20,000 new.</p>\n');
        }
    },
    'Render $...$ in text': {
        topic: md.render('Equation $x + y$.'),
        'Starts with "<p>Equation "': function(topic) {
            assert.isTrue(topic.startsWith('<p>Equation '));
        },
        'Ends with ".</p>"': function(topic) {
            assert.isTrue(topic.endsWith('</span>.</p>\n'));
        },
        'Contains math span': function(topic) {
            assert.notEqual(topic.indexOf('<span class="katex">'), -1);
        }
    },
    'Render $$...$$ in text': {
        topic: md.render('Before $$x + y$$ after.'),
        'Starts with "<p>Before "': function(topic) {
            assert.isTrue(topic.startsWith('<p>Before '));
        },
        'Ends with "after.</p>"': function(topic) {
            assert.isTrue(topic.endsWith('</span> after.</p>\n'));
        },
        'Contains math span': function(topic) {
            assert.notEqual(topic.indexOf('<span class="katex-display">'), -1);
        }
    }
}).export(module);
