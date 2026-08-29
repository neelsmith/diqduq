# Overview

`diqduq` is a python package leveraging language models with `dspy` to analyze the syntax of passages of Biblical Hebrew. The unique analytical scheme is specific to Biblical Hebrew, and is documented here.

In this scheme, analysis of a passage of Biblical Hebrew is expressed in two related structures:

- a list of verbal expressions, generally corresponding to clauses in an English translation
- a token-level table capturing principal relations in a dependency graph



## Table of verbal expressions

"Verbal expressions" are subject-verb ideas that most frequently correspond to clauses in an English translation. (Of course in Hebrew the subject may be implicit where that is not possible in English.) 

In the first draft of this scheme, we identify the following verb forms as *verbal expressions*.

1. *Every finite verb* constitutes a verbal expression. 
2. *Participles* constitute a verbal expression.

In this scheme, verbal expressions are classified according to:

1. their *syntactic type*. The possibilities for each construction are:
    - *independent* (also called "main" or "principal") verbs. These are syntactically independent finite verbs: their clause is syntacitcally coherent by itself.
    - *direct quote*: used for verbal expressions in directly quoted speech. 
2. by their *semantic type* ,as *transitive active*, *transitive passive*, *intransitive* or a *linking verb*.

`diqduq` recognizes as understood or implied:

1. elided present of `to be`: the verb "to be" is often elided. In this situation, a new token with unique ID must be added to the token table, and an entry added to the list of verbal expressions. The token will have `None` for its text value.  Example:  in לֹ֨א אֱלֹהִ֜ים הֵ֗מָּה, "they are not gods,"the subject is הֵמָּה, the predicate noun is אֱלֹהִים. We create a new token with `None` as its text value, and  enter its ID in the list of verbal expressions. This predicate construction is a main sentence, so its syntactic type will be *independent clause* and the semantic type will be *linking verb*.
 
 
## Token-level table of dependencies

### Tokenization
The analyzing program will keep track of the citation references for each token.. 

The text of the passage must be tokenized classified by its orthography as one of the following:

- *cantillation* token. Any of the *te'amim*. Examples: the *sof pasuq* `:`, *silluq*, *atnach*, among others.
- a *paragraph* token פ (*petuhah*) or ס (*setumah*) used to mark semantic divisions of the text.
- an *enclitic pronoun* token (as an object with a preposition or verb, or as a possessive with a noun). 
- the *proclitic conjunction* וְ
- the *joining token* ־ *maqaf *
-  a *lexical* token. These include continous sequences of alphabetic characters, vowel points (*niqqud*) on a word, *dagesh*, *mappiq*, *sin/shin* dot, but not cantillation marks (*te'amim*).
-  an *editiorial* token. Any Unicode punctuation character or other editorial marks such as the *masora* circle.

Example: the following sentence from the Masoretic text
 וַיְקַדֵּ֖שׁ אֹתֹ֑ו כִּ֣י בֹ֤ו שָׁבַת֙ מִכָּל־מְלַאכְתֹּ֔ו
begins with the *proclitic conjunction* וְ, then a lexical token יְקַדֵּשׁ. The next lexical token is אֹתֹו, which is an independent pronoun, not an enclitic form.  Contrast the string בֹ֤ו which contains two lexical tokens: the preposition בְּ and the third person singular object pronoun . In the phrase מִכָּל־מְלַאכְתֹּ֔ו, a lexical token מִכָּל is followed by the *joining token* ־, which is followed by two tokens, מְלַאכְתֹּו, containing a lexical token with a possessive object pronoun.


### Syntactic relations among tokens

We record the following set of relations among tokens.


#### Verbs and their principal construction

- verb of an independent clause: the `relation1` of independent verbs has the special value `root` which must not be used as the identifier for any token. Its `relationship1` value is `unit verb`. Example: in בְּרֵאשִׁ֖ית בָּרָ֣א אֱלֹהִ֑ים אֵ֥ת הַשָּׁמַ֖יִם וְאֵ֥ת הָאָֽרֶץ׃
there is an independent verb בָּרָ֣א with `relation1` value `root`, and `relationship1` value `unit verb`.

- verbs in direct quotes: the `relation1` will be the ID of the verb of the governing verbal expression, with a value of `direct quote` for `relationship1`. Example of direct quote:
in וַיֹּ֥אמֶר אֱלֹהִ֖ים יְהִ֣י אֹ֑ור וַֽיְהִי־אֹֽור׃
the  verbal unit anchored to יְהִ֣י is direct speech subordinate to יֹּ֥אמֶר. The token יְהִ֣י  will therefore have the id of  יֹּ֥אמֶר for its `relation1`, with `direct quote` as its `relationship1`.


- noun or pronoun serving as the subject of a verbal expression: *relation1* will be the id of the token of the verb.  The value of *relationship1* will be *subject*. Example: in *Genesis* 1.1,  בְּרֵאשִׁ֖ית בָּרָ֣א אֱלֹהִ֑ים אֵ֥ת הַשָּׁמַ֖יִם וְאֵ֥ת הָאָֽרֶץ׃, the explicit subject אֱלֹהִים has as `relation1` the id of the verb בָּרָא with `relationship1` value *subject*.

- noun or pronoun functioning as direct object of a verbal expression: *relation1* will be the id of the token of the verb. The value of *relationship1* will be *direct object*. In *Genesis* 1.1, each of the nouns שָּׁמַיִם and אָֽרֶץ will have the id of בָּרָא for `relation1` with *direct object* as the value of `relationship1`.

- noun or pronoun functioning as the predicate of a linking verb: *relation1* will be the id of the token of the verb, with *predicate* as the value of `relationship1`.


#### Coordinating conjunction
- coordinating conjunctions: when coordinating conjunctions join pairs of adjectives, nouns, prepositional phrases, or verbs, they use the IDs of the nouns, adjectives, prepositions of the prepositional phrases, or verb for `relation1` and `relation2`, and `coordinating conjunction` for both `relationship1` and `relationship2`. Example:  in בְּרֵאשִׁ֖ית בָּרָ֣א אֱלֹהִ֑ים אֵ֥ת הַשָּׁמַ֖יִם וְאֵ֥ת הָאָֽרֶץ׃, the conjunction וְ will have the ids of שָּׁמַ֖יִם and אָֽרֶץ and  for`relation1` and `relation2` and `coordinating conjunction` for both `relationship1` and `relationship2`. 

Note that the conjunction וְ  ( וַ ) can be repeated to form connecting pairs or even series of conjunctions.


We annotate this construction a little differently. The first connector is given the id the connected item as `relation1` and the id of the next connecting word as `relation2`. The second and following connectors take the id of the item they connect as `relation1` and the preceding connector as `relation2`. Example: in וַיְבָ֤רֶךְ אֱלֹהִים֙ אֶת־יֹ֣ום הַשְּׁבִיעִ֔י וַיְקַדֵּ֖שׁ אֹתֹ֑ו
We annotate this construction a little differently. The first connector is given the id the connected item as `relation1` and the id of the next connecting word as `relation2`. The second and following connectors take the id of the item they connect as `relation1` and the preceding connector as `relation2`. Example: in וַיְבָ֤רֶךְ אֱלֹהִים֙ אֶת־יֹ֣ום הַשְּׁבִיעִ֔י וַיְקַדֵּ֖שׁ אֹתֹ֑ו
there are two verbal expressions coordinated by וְ, namely וַיְבָ֤רֶךְ and וַיְקַדֵּ֖שׁ.  The first וְ (in וַיְבָ֤רֶךְ) takes the id of בָ֤רֶךְ
 for `relation1`, and the id of the second וְ  as `relation2`. The second וְ  (in  וַיְקַדֵּ֖שׁ)  takes the id of  קַדֵּ֖שׁ
 for `relation1`, and the id of the first וְ  as `relation2`. All have relationship values of `ccoordinating conjunction`.



### Noun relations, the article, adjectives

- other noun relations: when a noun or pronoun functions as the object of a preposition, it takes the ID of the preposition as `relation1` with `object of preposition` as the value of  `relationship1`. Example: in the phrase בְּאֶרֶץ , the noun אֶרֶץ is the object of the preposition  בְּ. It will have the ID of  בְּ for `relation1` with `object of preposition` for `relationship1`.

- the article: when the article  הַ  relates to a noun or adjective, it takes the id of the noun or adjective as `relation1` with `article` as the value of `relationship1`. Example: in הַשָּׁמַ֖יִם, the article has the ID of the noun הַשָּׁמַיִם for `relation1` and `article` for `relationship1`.

- the construct relation:  when two nouns stand in construct relation, the governing noun is recorded according to its function in the sentence. The related noun takes the ID of the governing noun as `relation1` with `construct` as the value of `relationship1`. Example:  in בְּזֵעַ֤ת אַפֶּ֙יךָ֙ תֹּ֣אכַל לֶ֔חֶם, the noun זֵעַת governs the the noun + possessive proun אַפֶּיךָ in a construct relationship. זֵעַת will be recorded as object of the preposition  בְּ (see below).  The noun אַפֶּי will have the id of זֵעַת as `relation1` and `construct` as the value of `reationship1`.
 
- adjectives: adjectives take the ID of the noun they modify as `relation1` with `adjectival` as the value of  `relationship1`. Example: in the phrase אֲחִיכֶם הַקָּטֹן, the adjective קָּטֹן modifies the noun אֲחִי (which is followed by the possessive pronoun). קָּטֹן will have the ID of אֲחִי for `relation1` and `adjectival` as the value of `relationship1`.
 

 

## TBA

- functoins of prepositions
- subordinating conjunctions: 
- The relative pronoun אֲשֶׁ֥ר : 








