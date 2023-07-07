
---
title: "Character Encoding"
excerpt: "Need and Methods"
categories:
  - IT
tags:
  - IT
  - Python

toc: true
---

You've written some interesting code, but all of them have processed only one kind of data - numbers. As you know (you can see this everywhere around you) lots of computer data are not numbers: first names, last names, addresses, titles, poems, scientific papers, emails, court judgments, love confessions, and much, much more. All these data must be stored, input, output, searched, and transformed by contemporary computers just like any other data, no matter if they are single characters or multi-volume encyclopedias.

##### How is it possible?

How can you do it in Python? This is what we'll discuss now. Let's start with how computers understand single characters.

Computers store characters as numbers. Every character used by a computer corresponds to a unique number, and vice versa. This assignment must include more characters than you might expect. Many of them are invisible to humans but essential to computers. Some of these characters are called whitespaces, while others are named control characters because their purpose is to control input/output devices. An example of a whitespace that is completely invisible to the naked eye is a special code, or a pair of codes (different operating systems may treat this issue differently), which are used to mark the ends of the lines inside text files.

People do not see this sign (or these signs) but are able to observe the effect of their application where the lines are broken. We can create virtually any number of character-number assignments, but life in a world in which every type of computer uses a different character encoding would not be very convenient. This system has led to a need to introduce a universal and widely accepted standard implemented by (almost) all computers and operating systems all over the world.

##### ASCII 

The one named ASCII (short for American Standard Code for Information Interchange) is the most widely used, and you can assume that nearly all modern devices (like computers, printers, mobile phones, tablets, etc.) use that code. The code provides space for 256 different characters, but we are interested only in the first 128. If you want to see how the code is constructed, look at [this table](https://www.cs.cmu.edu/~pattis/15-1XX/common/handouts/ascii.html). There are some interesting facts. Look at the code of the most common character - the space. This is 32.

Now check the code of the lowercase letter a. This is 97. And now find the upper-case A. Its code is 65. Now work out the difference between the code of a and A. It is equal to 32. That's the code of a space. Interesting, isn't it? Also, note that the letters are arranged in the same order as in the Latin alphabet.

##### I18N

Of course, the Latin alphabet is not sufficient for the whole of mankind. Users of that alphabet are in the minority. It was necessary to come up with something more flexible and capacious than ASCII, something able to make all the software in the world amenable to internationalization, because different languages use completely different alphabets, and sometimes these alphabets are not as simple as the Latin one.  The word internationalization is commonly shortened to I18N.

> I18N Internationalization

Why? Look carefully - there is an I at the front of the word, next, there are 18 different letters, and an N at the end. Despite the slightly humorous origin, the term is officially used in many documents and standards.
The software I18N is a standard in present times. Each program has to be written in a way that enables it to be used all around the world, among different cultures, languages, and alphabets.
A classic form of ASCII code uses eight bits for each sign. Eight bits mean 256 different characters. The first 128 are used for the standard Latin alphabet (both upper-case and lower-case characters). Is it possible to push all the other national characters used around the world into the remaining 128 locations? No. It isn't.

##### Code points and code pages

We need a new term now: a code point.

A code point is a number that makes a character. For example, 32 is a code point that makes a space in ASCII encoding. We can say that standard ASCII code consists of 128 code points.
As standard ASCII occupies 128 out of 256 possible code points, you can only make use of the remaining 128.  It's not enough for all possible languages, but it may be sufficient for one language, or for a small group of similar languages. Can you set the higher half of the code points differently for different languages? Yes, you can. Such a concept is called a code page.
A code page is a standard for using the upper 128 code points to store specific national characters. For example, there are different code pages for Western Europe and Eastern Europe, Cyrillic and Greek alphabets, Arabic and Hebrew languages, and so on. This means that the one and same code point can make different characters when used on different code pages.
For example, code point 200 makes Č (a letter used by some Slavic languages) when utilized by the ISO/IEC 8859-2 code page, and makes Ш (a Cyrillic letter) when used by the ISO/IEC 8859-5 code page.
In consequence, to determine the meaning of a specific code point, you have to know the target code page. In other words, the code points derived from the code page concept are ambiguous.

##### Unicode

Code pages helped the computer industry to solve I18N issues for some time, but it soon turned out that they would not be a permanent solution. The concept that solved the problem in the long term was Unicode.

Unicode assigns unique (unambiguous) characters (letters, hyphens, ideograms, etc.) to more than a million code points. The first 128 Unicode code points are identical to ASCII, and the first 256 Unicode code points are identical to the ISO/IEC 8859-1 code page (a code page designed for Western European languages).

###### UCS-4

The Unicode standard says nothing about how to code and store the characters in the memory and files. It only names all available characters and assigns them to planes (a group of characters of similar origin, application, or nature). There is more than one standard describing the techniques used to implement Unicode in actual computers and computer storage systems. The most general of them is UCS-4.

The name comes from Universal Character Set.

UCS-4 uses 32 bits (four bytes) to store each character, and the code is just the Unicode code points' unique number. A file containing UCS-4 encoded text may start with a BOM (byte order mark), an unprintable combination of bits announcing the nature of the file's contents. Some utilities may require it. As you can see, UCS-4 is a rather wasteful standard - it increases a text's size by four times compared to standard ASCII. Fortunately, there are smarter forms of encoding Unicode texts.

###### UTF-8

One of the most commonly used is UTF-8. The name is derived from Unicode Transformation Format. The concept is very smart. UTF-8 uses as many bits for each of the code points as it really needs to represent them.

For Example:

All Latin characters (and all standard ASCII characters) occupy eight bits; non-Latin characters occupy 16 bits; CJK (China-Japan-Korea) ideographs occupy 24 bits.
Due to features of the method used by UTF-8 to store the code points, there is no need to use the BOM, but some of the tools look for it when reading the file, and many editors set it up during the save.

Python 3 fully supports Unicode and UTF-8. You can use Unicode/UTF-8 encoded characters to name variables and other entities; you can use them during all input and output. This means that Python3 is completely I18Ned.

#### Other Encoding Schemes:

######  ANSI/Windows-1252

When the Windows operating system emerged in 1985, a new standard was quickly adopted known as the ANSI character set. The phrase “ANSI” was also known as the Windows code pages (Code Page 1252), even though it had nothing to do with the American National Standards Institute. Windows-1252 or CP-1252 (code page 1252) character encoding became popular with the advent of Microsoft Windows but was eventually superseded when Unicode was implemented within Windows. Unicode, which was first released in 1991, assigns a universal code to every character and symbol for all the languages in the world.

###### ISO-8859-1

The ISO-8859-1 (also known as Latin-1) character encoding set features all the characters of Windows-1252, including an extended subset of punctuation and business symbols. This standard was easily transportable across multiple word processors and even newly released versions of HTML 4. The first edition was published in 1987 and was a direct extension of the ASCII character set. While support was extensive for its time, the format was still limited.

#### Why is Character Encoding Important?

So it’s clear that each character set uses a unique table of identification codes to present a specific character to a user. If you were using the ISO-8859-1 character set to edit a document and then saved that document as a UTF-8 encoded document without declaring that the content was UTF-8, special characters and business symbols will render unreadable. Most modern web browsers support legacy character encodings, so a website can contain pages encoded in ISO-8859-1, or Windows-1252, or any other type of encoding. The browser should properly render those characters based on the character encoding format not being reported by the server.

However, if the character set is not correctly declared at the time the page is rendered, the web server’s default is usually to fall back without any specific character encoding format (usually ASCII).
This forces your browser or mobile device to determine the page’s proper type of character encoding. Based on the WHATWG specifications adopted by W3C, the most typical default fallback is UTF-8. However, some browsers will fall back to ASCII.

Proper character encoding is vital for properly rendering online text and plays a critical role in localization projects.
