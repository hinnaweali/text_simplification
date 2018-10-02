# coding=utf-8
import xml.etree.ElementTree as etree
from collections import Counter
import math
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import tokenize
import time
import json
from wiki2plain import Wiki2Plain
import collections
from thesaurus import Word
import os
from nltk.tag.stanford import StanfordPOSTagger
from pattern3.text import singularize, pluralize, keywords, Parser
from pattern3.text.en.inflect import Verbs
import pickle
from textblob import TextBlob

special_list = ['url', 'html', 'iii', 'xiii', 'vii', 'iiv', 'xxi', 'iix', 'xxv', 'ixx', 'xvi', 'vvi', 'ivv','iis', 'ascii',
                'wii' ,'iis','=','gif','jpg','jpeg','png','bmp','http']
stop_words_basic_words = ['are', 'been', 'can', 'could', 'did', 'first', 'had', 'has', 'her', 'him', 'his', 'into',
                          'its', 'know', 'man', 'men', 'more', 'must', 'one', 'our', 'she', 'should', 'their', 'them',
                          'these', 'they', 'time', 'two', 'upon', 'was', 'were', 'what', 'which', 'would', 'your',
                          'back', "can't", "didn't", "don't", 'good', "he's", 'hey', "i'll", "i'm", "it's", 'just',
                          'like', 'look', 'mean', 'okay', 'really', 'right', 'something', 'tell', "that's", 'yeah',
                          "you're", 'above', 'alone', 'along', 'already', 'also', 'although', 'always', 'another',
                          'anybody', 'anyone', 'anything', 'anywhere', 'area', 'areas', 'around', 'ask', 'asked',
                          'asking', 'asks', 'away', 'backed', 'backing', 'backs', 'became', 'become', 'becomes',
                          'began', 'behind', 'being', 'beings', 'best', 'better', 'big', 'both', 'came', 'cannot',
                          'case', 'cases', 'certain', 'certainly', 'clear', 'clearly', 'differ', 'different',
                          'differently', 'does', 'done', 'downed', 'downing', 'downs', 'during', 'each', 'early',
                          'either', 'end', 'ended', 'ending', 'ends', 'evenly', 'everybody', 'everyone', 'everything',
                          'everywhere', 'faces', 'fact', 'facts', 'felt', 'few', 'find', 'finds', 'four', 'full',
                          'fully', 'further', 'furthered', 'furthering', 'furthers', 'gave', 'general', 'generally',
                          'gets', 'given', 'gives', 'going', 'goods', 'got', 'great', 'greater', 'greatest', 'group',
                          'grouped', 'grouping', 'groups', 'having', 'herself', 'high', 'higher', 'highest', 'himself',
                          'however', 'important', 'interest', 'itself', 'kind', 'knew', 'known', 'knows', 'large',
                          'largely', 'last', 'later', 'latest', 'least', 'less', 'lets', 'likely', 'long', 'longer',
                          'longest', 'made', 'making', 'many', 'member', 'members', 'might', 'most', 'mostly', 'mrs',
                          'myself', "n't", 'necessary', 'need', 'needed', 'needing', 'needs', 'never', 'new', 'newer',
                          'newest', 'next', 'nobody', 'non', 'noone', 'nothing', 'nowhere', 'number', 'numbers',
                          'often', 'old', 'older', 'oldest', 'once', 'open', 'opened', 'opening', 'opens', 'order',
                          'ordered', 'ordering', 'orders', 'others', 'part', 'parted', 'parting', 'parts', 'per',
                          'perhaps', 'place', 'places', 'point', 'pointed', 'pointing', 'points', 'possible', 'present',
                          'problem', 'problems', 'puts', 'rather', 'room', 'rooms', 'said', 'same', 'saw', 'says',
                          'second', 'seconds', 'seemed', 'seeming', 'seems', 'sees', 'several', 'shall', 'show',
                          'showed', 'showing', 'shows', 'side', 'sides', 'since', 'small', 'smaller', 'smallest',
                          'somebody', 'someone', 'somewhere', 'state', 'states', 'sure', 'taken', 'therefore', 'thing',
                          'things', 'think', 'thinks', 'those', 'thought', 'thoughts', 'three', 'thus', 'today', 'too',
                          'took', 'toward', 'turn', 'turned', 'turning', 'turns', 'until', 'use', 'used', 'uses',
                          'want', 'wanted', 'wanting', 'wants', 'way', 'ways', 'wells', 'went', 'whether', 'whole',
                          'whose', 'within', 'without', 'work', 'year', 'years', 'yet', 'young', 'younger', 'youngest',
                          'yours', '-lrb-', '-rrb-', '-lsb-', '-rsb-', '...', "aren't", 'below', "couldn't", "doesn't",
                          'doing', "hadn't", "hasn't", "haven't", "he'd", "he'll", "here's", 'hers', "how's", "i'd",
                          "i've", "isn't", "let's", "mustn't", 'nor', 'ought', 'ours', 'ourselves', 'own', "shan't",
                          "she'd", "she'll", "she's", "shouldn't", 'theirs', 'themselves', "there's", "they'd",
                          "they'll", "they're", "they've", "wasn't", "we'd", "we'll", "we're", "we've", "weren't",
                          "what's", "when's", "where's", "who's", 'whom', "why's", "won't", "wouldn't", "you'd",
                          "you'll", "you've", 'yourself', 'yourselves', 'return', 'arent', 'cant', 'couldnt', 'didnt',
                          'doesnt', 'dont', 'hadnt', 'hasnt', 'havent', 'hes', 'heres', 'hows', 'isnt', 'mustnt',
                          'shant', 'shes', 'shouldnt', 'thats', 'theres', 'theyll', 'theyre', 'theyve', 'wasnt',
                          'werent', 'whats', 'whens', 'wheres', 'whos', 'whys', 'wont', 'wouldnt', 'youd', 'youll',
                          'youre', 'youve', "that'll", 'don', "should've", 'ain', 'aren', 'couldn', 'didn', 'doesn',
                          'hadn', 'hasn', 'haven', 'isn', 'mightn', "mightn't", 'mustn', 'needn', "needn't", 'shan',
                          'aboard', 'alongside', 'amid', 'amidst', 'amongst', 'anti', 'astride', 'aught', 'bar',
                          'barring', 'beneath', 'beside', 'besides', 'beyond', 'circa', 'concerning', 'considering',
                          "daren't", 'despite', 'except', 'excepting', 'excluding', 'fewer', 'five', 'following',
                          'goes', 'hisself', 'idem', 'ilk', 'including', 'inside', 'mine', 'minus', 'naught', 'neither',
                          'none', 'notwithstanding', 'oneself', 'onto', 'opposite', 'otherwise', "oughtn't", 'ourself',
                          'outside', 'past', 'pending', 'plus', 'regarding', 'round', 'save', 'seen', 'self',
                          'somewhat', 'suchlike', 'sundry', 'thee', 'thine', 'thou', 'throughout', 'thyself', 'tother',
                          'towards', 'twain', 'underneath', 'unless', 'unlike', 'various', 'versus', 'via', 'visavis',
                          'whatall', 'whatever', 'whatsoever', 'whereas', 'wherewith', 'wherewithal', 'whichever',
                          'whichsoever', 'whoever', 'whomever', 'whomso', 'whomsoever', 'whosoever', 'worth', 'yon',
                          'yonder', 'youall', 'sunday', 'southeast', 'fourth', '3rd', 'fifth', 'noon', "i'll",
                          'shouldn', 'wouldn', '1850s', '1860s', '1870s', '1880s', '1890s', '1830s', '1832s', 'begin',
                          'america', 'jan', 'feb', 'mar', 'apr', 'aug', 'sept', 'nov', 'dec', 'six', 'seven', 'eight',
                          'nine', 'tene', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen',
                          'eighteen', 'ninteen', 'twenty', 'hundred', 'thousand', 'milion', 'days', 'day', 'date',
                          'monday', 'tuesday', 'wednesday', 'thursady', 'friday', 'saturday', 'january', 'february',
                          'march', 'april', 'june', 'july', 'augest', 'september', 'october', 'november', 'december',
                          'hello', 'life', 'mobile', 'paper', 'usb', 'chair', 'autumn', 'winter', 'summer', 'season',
                          'white', 'yellow', 'green', 'blue', 'black', 'red', 'lila', 'violet', 'brown', 'cream', 'bit',
                          'byte', 'nice', 'uncle', 'months', 'begins', 'brother', 'father', 'mother', 'sister', 'night',
                          'evenning', 'month', 'week', 'monady', 'thursday', 'oct', "i'll", "i'm", "i'll", "i'll",
                          "i'm"]
countries_list = ['usa', 'afghanistan', 'albania', 'algeria', 'american samoa', 'andorra', 'angola', 'anguilla',
                  'antigua  barbuda', 'argentina', 'armenia', 'aruba', 'australia', 'austria', 'azerbaijan',
                  'bahamas, the', 'bahrain', 'bangladesh', 'barbados', 'belarus', 'belgium', 'belize', 'benin',
                  'bermuda', 'bhutan', 'bolivia', 'bosnia  herzegovina', 'botswana', 'brazil', 'british virgin is.',
                  'brunei', 'bulgaria', 'burkina faso', 'burma', 'burundi', 'cambodia', 'cameroon', 'canada',
                  'cape verde', 'cayman islands', 'central african rep.', 'chad', 'chile', 'china', 'colombia',
                  'comoros', 'congo, dem. rep.', 'congo, repub. of the', 'cook islands', 'costa rica', 'cote d',
                  'ivoire', 'croatia', 'cuba', 'cyprus', 'czech republic', 'denmark', 'djibouti', 'dominica',
                  'dominican republic', 'east timor', 'ecuador', 'egypt', 'el salvador', 'equatorial guinea', 'eritrea',
                  'estonia', 'ethiopia', 'faroe islands', 'fiji', 'finland', 'france', 'french guiana',
                  'french polynesia', 'gabon', 'gambia, the', 'gaza strip', 'georgia', 'germany', 'ghana', 'gibraltar',
                  'greece', 'greenland', 'grenada', 'guadeloupe', 'guam', 'guatemala', 'guernsey', 'guinea',
                  'guineabissau', 'guyana', 'haiti', 'honduras', 'hong kong', 'hungary', 'iceland', 'india',
                  'indonesia', 'iran', 'iraq', 'ireland', 'isle of man', 'israel', 'italy', 'jamaica', 'japan',
                  'jersey', 'jordan', 'kazakhstan', 'kenya', 'kiribati', 'korea, north', 'korea, south', 'kuwait',
                  'kyrgyzstan', 'laos', 'latvia', 'lebanon', 'lesotho', 'liberia', 'libya', 'liechtenstein',
                  'lithuania', 'luxembourg', 'macau', 'macedonia', 'madagascar', 'malawi', 'malaysia', 'maldives',
                  'mali', 'malta', 'marshall islands', 'martinique', 'mauritania', 'mauritius', 'mayotte', 'mexico',
                  'micronesia, fed. st.', 'moldova', 'monaco', 'mongolia', 'montserrat', 'morocco', 'mozambique',
                  'namibia', 'nauru', 'nepal', 'netherlands', 'netherlands antilles', 'new caledonia', 'new zealand',
                  'nicaragua', 'niger', 'nigeria', 'n. mariana islands', 'norway', 'oman', 'pakistan', 'palau',
                  'panama', 'papua new guinea', 'paraguay', 'peru', 'philippines', 'poland', 'portugal', 'puerto rico',
                  'qatar', 'reunion', 'romania', 'russia', 'rwanda', 'saint helena', 'saint kitts  nevis',
                  'saint lucia', 'st pierre  miquelon', 'saint vincent and the grenadines', 'samoa', 'san marino',
                  'sao tome  principe', 'saudi arabia', 'senegal', 'serbia', 'seychelles', 'sierra leone', 'singapore',
                  'slovakia', 'slovenia', 'solomon islands', 'somalia', 'south africa', 'spain', 'sri lanka', 'sudan',
                  'suriname', 'swaziland', 'sweden', 'switzerland', 'syria', 'taiwan', 'tajikistan', 'tanzania',
                  'thailand', 'togo', 'tonga', 'trinidad  tobago', 'tunisia', 'turkey', 'turkmenistan', 'tuvalu',
                  'uganda', 'ukraine', 'united arab emirates', 'united kingdom', 'united states', 'uruguay',
                  'uzbekistan', 'vanuatu', 'venezuela', 'vietnam', 'virgin islands', 'wallis and futuna', 'west bank',
                  'western sahara', 'yemen', 'zambia', 'zimbabwe']
cities_list = ['Abuja', 'Accra', 'Adamstown', 'Addis Ababa', 'Algiers', 'Alofi', 'Amman', 'Amsterdam',
               'Andorra la Vella', 'Ankara', 'Antananarivo', 'Apia', 'Ashgabat', 'Asmara', 'Astana', 'Athens', 'Avarua',
               'Baghdad', 'Baku', 'Bamako', 'Bandar Seri Begawan', 'Bangkok', 'Bangui', 'Banjul', 'Basseterre',
               'Beijing', 'Beirut', 'Belgrade', 'Belmopan', 'Berlin', 'Bern', 'Bishkek', 'Bissau', 'Bogot ', 'Bras lia',
               'Bratislava', 'Brazzaville', 'Bridgetown', 'Brussels', 'Bucharest', 'Budapest', 'Buenos Aires',
               'Bujumbura', 'Cairo', 'Canberra', 'Caracas', 'Castries', 'Cayenne', 'Charlotte Amalie', 'Chisinau',
               'Cockburn Town', 'Conakry', 'Copenhagen', 'Dakar', 'Damascus', 'Dhaka', 'Dili', 'Djibouti', 'Dodoma',
               'Dar es Salaam', 'Doha', 'Douglas', 'Dublin', 'Dushanbe', 'Edinburgh of the Seven Seas',
               'Episkopi Cantonment', 'Flying Fish Cove', 'Freetown', 'Funafuti', 'Gaborone', 'George Town',
               'Georgetown', 'Gibraltar', 'Grytviken', 'Guatemala City', 'Hamilton', 'Hanga', 'Roma', 'Hanoi', 'Harare',
               'Hargeisa', 'Havana', 'Helsinki', 'Honiara', 'Islamabad', 'Jakarta', 'Jamestown', 'Jerusalem', 'Juba',
               'Kabul', 'Kampala', 'Kathmandu', 'Khartoum', 'Kiev', 'Kigali', 'Kingston', 'Kingstown', 'Kinshasa',
               'Kuala Lumpur', 'Putrajaya', 'Kuwait City', 'Libreville', 'Lilongwe', 'Lima', 'Lisbon', 'Ljubljana',
               'Lom ', 'London', 'Luanda', 'Lusaka', 'Luxembourg', 'Madrid', 'Majuro', 'Malabo', 'Mal ', 'Managua',
               'Manama', 'Manila', 'Maputo', 'Marigot', 'Maseru', 'Mata-Utu', 'Mbabane ', 'Lobamba', 'Melekeok',
               'Ngerulmud', 'Mexico City', 'Minsk', 'Mogadishu', 'Monaco', 'Monrovia', 'Montevideo', 'Moroni', 'Moscow',
               'Muscat', 'Nairobi', 'Nassau', 'Naypyidaw', "N'Djamena", 'New Delhi', 'Niamey', 'Nicosia', 'Nouakchott',
               'Nukualofa', 'Nuuk', 'Oranjestad', 'Oslo', 'Ottawa', 'Ouagadougou', 'Pago Pago', 'Palikir',
               'Panama City', 'Papeete', 'Paramaribo', 'Paris', 'Philipsburg', 'Phnom Penh', 'Plymouth',
               'Brades Estate', 'Podgorica', 'Cetinje', 'Port Louis', 'Port Moresby', 'Port Vila', 'Port of Spain',
               'Cotonou', 'Prague', 'Praia', 'Pretoria', 'Bloemfontein', 'Cape Town', 'Pristina', 'Pyongyang', 'Quito',
               'Rabat', 'Reykjav k', 'Riga', 'Riyadh', 'Road Town', 'Rome', 'Roseau', 'Saipan', 'San Jose', 'San Juan',
               'San Marino', 'San Salvador', "Sana'a", 'Santiago ', 'Valparaiso', 'Santo Domingo', 'Sarajevo', 'Seoul',
               'Singapore', 'Skopje', 'Sofia', 'Sri Jayawardenepura Kotte', 'Colombo ', 'St. Helier', 'St. Peter Port',
               'St. Pierre', 'Stanley', 'Stepanakert', 'Stockholm', 'Sucre', 'La Paz', 'Sukhumi', 'Suva', 'Taipei',
               'Tallinn', 'Tarawa', 'Tashkent', 'Tbilisi ', 'Kutaisi', 'Tegucigalpa', 'Tehran', 'Thimphu', 'Tirana',
               'Tiraspol', 'Tokyo', 'T rshavn', 'Tripoli', 'Tskhinvali', 'Tunis', 'Ulan Bator', 'Vaduz', 'Valletta',
               'The Valley', 'Vatican City', 'Victoria', 'Vienna', 'Vientiane', 'Vilnius', 'Warsaw', 'Washington',
               'Wellington', 'West Island', 'Willemstad', 'Windhoek', 'Yamoussoukro', 'Abidjan', 'Yaound', 'Yaren',
               'Yerevan', 'alexandria', 'algiers', 'almaty', 'ankara', 'antananarivo', 'antwerp', 'athens', 'bangkok',
               'banjul', 'beirut', 'belgrade', 'bombay', 'bratislava', 'brussels', 'bucharest', 'cairo', 'calcutta',
               'cape town', 'cologne', 'copenhagen', 'dacca', 'damascus', 'dublin', 'florence', 'foochow', 'genoa',
               'geneva', 'harare', 'helsinki', 'ho chi minh city', 'istanbul', 'jakarta', 'kiev', 'kinshasa',
               'kuwait city', 'lisbon', 'lubumbashi', 'lyons', 'madras', 'maputo', 'mexico city', 'milan', 'moscow',
               'munich', 'nanking', 'naples', 'nuuk', 'oslo', 'peking', 'prague', 'pusan', 'quebec city', 'rangoon',
               'rome', 'sofia', 'saint petersburg', 'swatow', 'tallinn', 'tbilisi', 'the hague', 'tientsin', 'turin',
               'ulan bator', 'venice', 'vienna', 'vilnius', 'warsaw', 'berlin', 'paris', 'london', 'washington', 'roma']
others = ['edu','com','net','info','org','aaa' ,'bbb' ,'ccc' ,'ddd' ,'eee' ,'fff' ,'ggg' ,'hhh' ,'jjj' ,'kkk' ,'lll' ,'mmm' ,'nnn' ,'ooo' ,'ppp'
          ,'qqq' ,'rrr' ,'sss' ,'ttt' ,'vvv' ,'xxx' ,'yyy' ,'www' ,'zzz' ,'angle', 'ant', 'apple', 'arch', 'arm', 'army', 'chapter', 'baby', 'bag', 'ball', 'band', 'basin', 'basket',
          'bath', 'bed', 'bee', 'bell', 'berry', 'bird', 'blade', 'board', 'boat', 'bone', 'book', 'boot', 'bottle',
          'box', 'boy', 'brain', 'brake', 'branch', 'brick', 'bridge', 'brush', 'bucket', 'bulb', 'button', 'cake',
          'camera', 'card', 'cart', 'carriage', 'cat', 'chain', 'cheese', 'chest', 'chin', 'church', 'circle', 'clock',
          'cloud', 'coat', 'collar', 'comb', 'cord', 'cow', 'cup', 'curtain', 'cushion', 'dog', 'door', 'drain',
          'drawer', 'dress', 'drop', 'ear', 'egg', 'engine', 'eye', 'face', 'farm', 'feather', 'finger', 'fish', 'flag',
          'floor', 'fly', 'foot', 'fork', 'fowl', 'frame', 'garden', 'girl', 'glove', 'goat', 'gun', 'hair', 'hammer',
          'hand', 'hat', 'head', 'heart', 'hook', 'horn', 'horse', 'hospital', 'house', 'island', 'jewel', 'kettle',
          'key', 'knee', 'knife', 'knot', 'leaf', 'leg', 'library', 'line', 'lip', 'lock', 'map', 'match', 'monkey',
          'moon', 'mouth', 'muscle', 'nail', 'neck', 'needle', 'nerve', 'net', 'nose', 'nut', 'office', 'orange',
          'oven', 'parcel', 'pen', 'pencil', 'picture', 'pig', 'pin', 'pipe', 'plane', 'plate', 'plough/plow', 'pocket',
          'pot', 'potato', 'prison', 'pump', 'rail', 'rat', 'receipt', 'ring', 'rod', 'roof', 'root', 'sail', 'school',
          'scissors', 'screw', 'seed', 'sheep', 'shelf', 'ship', 'shirt', 'shoe', 'skin', 'skirt', 'snake', 'sock',
          'spade', 'sponge', 'spoon', 'spring', 'square', 'stamp', 'star', 'station', 'stem', 'stick', 'stocking',
          'stomach', 'store', 'street', 'sun', 'table', 'tail', 'thread', 'throat', 'thumb', 'ticket', 'toe', 'tongue',
          'tooth', 'town', 'train', 'tray', 'tree', 'trousers', 'umbrella', 'wall', 'watch', 'wheel', 'whip', 'whistle',
          'window', 'wing', 'wire', 'worm', 'come', 'get', 'give', 'keep', 'let', 'make', 'put', 'seem', 'take', 'have',
          'say', 'see', 'send', 'may', 'will', 'about', 'across', 'after', 'against', 'among', 'before', 'between',
          'down', 'from', 'off', 'over', 'through', 'under', 'with', 'for', 'till', 'than', 'the', 'all', 'any',
          'every', 'little', 'much', 'other', 'some', 'such', 'that', 'this', 'you', 'who', 'and', 'because', 'but',
          'though', 'while', 'how', 'when', 'where', 'why', 'again', 'ever', 'far', 'forward', 'here', 'near', 'now',
          'out', 'still', 'then', 'there', 'together', 'well', 'almost', 'enough', 'even', 'not', 'only', 'quite',
          'very', 'tomorrow', 'yesterday', 'north', 'south', 'east', 'west', 'please', 'yes']
person_names = ['nicholas' ,'juliana', 'nelson', 'hailey', 'anne', 'nelson', 'horatio', 'nero', 'neruda', 'pablo', 'newhart',
                'bob', 'newton', 'isaac', 'newton', 'john', 'nicks', 'stevie', 'czar', 'nicholas', 'ii', 'nicoll',
                'james', 'niebuhr', 'reinhold', 'niemller', 'martin', 'nietzsche', 'friedrich', 'nightingale',
                'florence', 'nijinsky', 'vaslav', 'nin', 'anas', 'nin', 'ninio', 'jacques', 'niranjan', 'sangeeta',
                'niven', 'larry', 'nixon', 'richard', 'noam', 'eli', 'norton', 'joshua', 'abraham', 'nostradamus',
                'novalis', 'nugent', 'ted', 'nukem', 'duke', 'null', 'gary', 'nunally', 'patrick', 'nuwas', 'abu',
                'obama', 'barack', 'oberst', 'conor', "o'brien", 'conan', 'ochs', 'phil', "o'donnell", 'rosie', 'ogi',
                'adolf', 'ogilvy', 'david', "o'keeffe", 'georgia', 'oliver', 'jamie', 'oliver', 'robert', 'olson',
                'ken', 'olsen', 'mary', 'kate', 'and', 'ashley', 'onassis', 'jacqueline', 'kennedy', 'ondrick',
                'william', 'oppenheimer', 'robert', "o'reilly", 'bill', "o'rourke", 'ortiz', 'david', 'orwell',
                'george', 'osama', 'bin', 'laden', 'osbourne', 'ozzy', 'oshaughnessy', 'arthur', 'osho', 'ouspensky',
                'overbury', 'thomas', 'ovid', 'owens', 'jesse', 'owsen', 'dan', 'pape', 'page', 'larry', 'paige',
                'satchel', 'paine', 'thomas', 'palahniuk', 'chuck', 'palgrave', 'francis', 'turner', 'palin', 'michael',
                'palme', 'olof', 'parker', 'dorothy', 'parton', 'dolly', 'pascal', 'blaise', 'pasteur', 'louis',
                'patajali', 'pater', 'walter', 'paterson', 'isabel', 'patrick', 'saint', 'patton', 'george', 'paul',
                'vi', 'paul', 'ron', 'pauli', 'wolfgang', 'pauling', 'linus', 'payack', 'paul', 'jj', 'payne', 'max',
                'peel', 'john', 'peguy', 'charles', 'peirce', 'charles', 'sanders', 'penn', 'william', 'percy',
                'walker', 'peres', 'shimon', 'perger', 'andreas', 'paolo', 'pericles', 'perle', 'richard', 'perlis',
                'alan', 'perry', 'michael', 'perry', 'oliver', 'hazard', 'pessoa', 'fernando', 'pet', 'shop', 'boys',
                'peter', 'kay', 'peter', 'dr', 'laurence', 'petronius', 'gaius', 'petty', 'tom', 'phelps', 'michael',
                'philip', 'duke', 'of', 'edinburgh', 'philips', 'emo', 'piaget', 'jean', 'picasso', 'pablo', 'pieper',
                'josef', 'pinker', 'steven', 'piper', 'roddy', 'piraten', 'fritiof', 'nilsson', 'pirsig', 'robert',
                'plath', 'sylvia', 'plato', 'plimpton', 'martha', 'poe', 'edgar', 'allan', 'poincar', 'henri', 'roman',
                'pompey', 'the', 'great', 'ponsonby', 'arthur', 'pope', 'alexander', 'popper', 'karl', 'porson',
                'richard', 'postel', 'jon', 'powell', 'colin', 'pratchett', 'terry', 'premchand', 'munshi', 'presley',
                'elvis', 'priestley', 'joseph', 'proudhon', 'puget', 'jade', 'punja', 'hari', 'purcell', 'steve',
                'pushkin', 'aleksandr', 'putin', 'vladimir', 'pynchon', 'thomas', 'qarase', 'laisenia', 'qin', 'shi',
                'huang', 'quale', 'anthony', 'quarles', 'francis', 'quayle', 'dan', 'marie', 'queen', 'of', 'romania',
                'quine', 'willard', 'van', 'orman', 'quintilian', 'marcus', 'fabius', 'quisenberry', 'dan', 'raajan',
                'amitrajit', 'rabelais', 'franois', 'rabin', 'yitzhak', 'rabuka', 'sitiveni', 'radner', 'gilda', 'rae',
                'pramod', 'raleigh', 'sir', 'walter', 'ramirez', 'manny', 'rand', 'ayn', 'ranke', 'leopold', 'von',
                'ransome', 'arthur', 'rascoe', 'burton', 'raskin', 'jef', 'ravuvu', 'asesela', 'ray', 'gene', 'ray',
                'james', 'arthur', 'reagan', 'nancy', 'reagan', 'ronald', 'reagan', 'ron', 'rees', 'nigel', 'reeve',
                'christopher', 'rehnquist', 'william', 'reich', 'wilhelm', 'reid', 'harry', 'reisman', 'george',
                'rexroth', 'kenneth', 'rice', 'condoleezza', 'richards', 'keith', 'richelieu', 'cardinal', 'rickover',
                'hyman', 'riley', 'tim', 'rilke', 'rainer', 'maria', 'rimbaud', 'arthur', 'ripa', 'kelly', 'ritter',
                'scott', 'rivers', 'joan', 'ro', 'robbins', 'anthony', 'robespierre', 'maximilien', 'robinson',
                'jackie', 'robinson', 'kim', 'stanley', 'robinson', 'spider', 'rockefeller', 'john', 'rockne', 'knute',
                'rodriguez', 'alex', 'rogers', 'fred', 'rogers', 'will', 'romano', 'ray', 'rommel', 'erwin',
                'roosevelt', 'eleanor', 'roosevelt', 'franklin', 'delano', 'roosevelt', 'theodore', 'rosing', 'wayne',
                'roth', 'geneen', 'roth', 'philip', 'rothbard', 'murray', 'rothschild', 'baron', 'roux', 'joseph',
                'rousseau', 'jean jacques', 'rovabokola', 'ratu', 'viliame', 'roy', 'arundhati', 'ru', 'rudner', 'rita',
                'ruiz', 'don', 'miguel', 'rukeyser', 'louis', 'rumbold', 'richard', 'rumi', 'jalal', 'al din',
                'muhammad', 'rumsfeld', 'donald', 'runnels', 'dustin', 'rushton', 'willie', 'ruskin', 'john', 'russell',
                'bertrand', 'russell', 'nipsey', 'russell', 'rosaland', 'ruth', 'babe', 'rutherford', 'ernest', 'sa se',
                'sbato', 'ernesto', 'sade', 'donatien', 'de', 'sagan', 'carl', 'saint exupry', 'antoine', 'de',
                'saint just', 'louis', 'de', 'saki', 'salk', 'jonas', 'salinger', 'sallust', 'sanger', 'margaret',
                'san', 'martn', 'jos', 'de', 'santayana', 'george', 'sappho', 'saroyan', 'william', 'sartre',
                'jean paul', 'sarven', 'allen', 'sathya', 'sai', 'baba', 'satriani', 'joe', 'sawyer', 'diane', 'saul',
                'john', 'ralston', 'savitri', 'devi', 'schaar', 'john', 'schama', 'simon', 'schiller', 'friedrich',
                'von', 'schlter', 'poul', 'schneier', 'bruce', 'schopenhauer', 'arthur', 'schrder', 'gerhard',
                'schulberg', 'budd', 'schultz', 'charles', 'schweitzer', 'albert', 'seabrook', 'jeremy', 'seagal',
                'steven', 'sedaris', 'david', 'sedgwick', 'john', 'seigle', 'lucy', 'seldes', 'george', 'sellers',
                'peter', 'serling', 'rod', 'serrano', 'miguel', 'seuss', 'dr', 'sh so', 'shahak', 'israel',
                'shakespeare', 'william', 'shakur', 'tupac', 'shankar', 'ravi', 'sharpton', 'al', 'shaw', 'george',
                'bernard', 'shawcross', 'hartley', 'sheck', 'barry', 'shedd', 'john', 'shelley', 'mary',
                'wollstonecraft', 'shelley', 'percy', 'bysshe', 'sheridan', 'richard', 'brinsley', 'sherman', 'william',
                'tecumseh', 'shevardnadze', 'eduard', 'shinoda', 'mike', 'shirley', 'james', 'schopenhauer', 'arthur',
                'shrivastava', 'mataji', 'nirmala', 'siegel', 'jerry', 'simak', 'clifford', 'simon', 'paul',
                'simonides', 'of', 'ceos', 'simpson', 'jack', 'simpson', 'jessica', 'sinclair', 'upton', 'singer',
                'isaac', 'bashevis', 'sidhu', 'navjot', 'singh', 'singh', 'raman', 'pratap', 'singh', 'sir', 'vijay',
                'sitwell', 'edith', 'sixtus', 'v', 'scanderbeg|skenderbeu', 'slick', 'grace', 'smirnoff', 'yakov',
                'smith', 'adam', 'smith', 'anna', 'nicole', 'smith', 'elliott', 'smith', 'gordon', 'smith', 'joseph',
                'smith', 'kevin', 'smith', 'logan', 'pearsall', 'smith', 'margaret', 'chase', 'smith', 'sydney',
                'smollett', 'tobias', 'smuts', 'jan', 'christiaan', 'snepscheut', 'jan', 'van', 'de', 'snoop', 'dogg',
                'snow', 'carrie', 'sobran', 'joseph', 'socrates', 'solomon', 'solzhenitsyn', 'alexander', 'sontag',
                'susan', 'soong', 'may ling', 'sorkin', 'aaron', 'soros', 'george', 'southerne', 'thomas', 'sowell',
                'thomas', 'sp sy', 'sparber', 'max', 'spears', 'britney', 'speijk', 'jan', 'van', 'spence', 'gerry',
                'spencer', 'herbert', 'spielberg', 'steven', 'spinoza', 'baruch', 'spock', 'benjamin', 'spolsky',
                'joel', 'springsteen', 'bruce', 'spurgeon', 'charles', 'haddon', 'stalin', 'joseph', 'stallman',
                'richard', 'stanley', 'henry', 'morton', 'stanton', 'elizabeth', 'cady', 'starr', 'ringo', 'starrett',
                'vincent', 'stein', 'gertrude', 'stein', 'herbert', 'steinbeck', 'john', 'steinem', 'gloria',
                'stephenson', 'neal', 'sterling', 'bruce', 'stevens', 'wallace', 'stevenson', 'adlai', 'stevenson',
                'robert', 'louis', 'stewart', 'james', 'stewart', 'jon', 'stein', 'ben', 'stimson', 'henry',
                'stokowski', 'leopold', 'stone', 'stone', 'lucy', 'stone', 'clement', 'stoner', 'jr', 'james',
                'stoppard', 'tom', 'stout', 'rex', 'stowe', 'madeleine', 'stratford', 'lord', 'strauss', 'richard',
                'stravinsky', 'igor', 'stroustrup', 'bjarne', 'sturgeon', 'theodore', 'sukuna', 'ratu', 'sir', 'lala',
                'sulla', 'lucius', 'cornelius', 'sun', 'tzu', 'ustauskas', 'vytautas', 'sutherland', 'kiefer',
                'suvorov', 'alexander', 'vasilyevich', 'suzuka', 'shunryu', 'swedenborg', 'emanuel', 'sweetnam', 'skye',
                'swift', 'jonathan', 'swinburne', 'charles', 'algernon', 'swindoll', 'charles', 'syrus', 'publilius',
                'sz', 'szilard', 'leo', 'ta to', 'tacitus', 'tagore', 'rabindranath', 'taleb', 'nassim', 'nicholas',
                'tannen', 'deborah', 'tate', 'sharon', 'tavola', 'kaliopate', 'tchaikovsky', 'pyotr', 'ilyich',
                'tecumseh', 'teilhard', 'de', 'chardin', 'pierre', 'tennyson', 'alfred', 'tenzin', 'gyatso', 'teresa',
                'of', 'avila', 'tertullian', 'tesla', 'nikola', 'thackeray', 'william', 'makepeace', 'thant', 'u',
                'thatcher', 'margaret', 'the', 'rock', 'theron', 'charlize', 'thomas', 'aquinas', 'thompson', 'dorothy',
                'thompson', 'hunter', 'thomson', 'william', 'lord', 'kelvin', 'thoreau', 'henry', 'david', 'throttle',
                'ben', 'thucydides', 'thurber', 'james', 'tilopa', 'tito', 'josip', 'broz', 'tocqueville', 'alexis',
                'de', 'todorov', 'tzvetan', 'tohei', 'koichi', 'tolkien', 'tolstoy', 'leo', 'tomlin', 'lily', 'tora',
                'apisai', 'torre', 'joe', 'torvalds', 'linus', 'toynbee', 'arnold', 'toynbee', 'arnold', 'joseph',
                'tr tw', 'traficant', 'james', 'jr', 'travaglia', 'simon', 'travers', 'trollope', 'anthony', 'trotsky',
                'leon', 'trudeau', 'pierre', 'truman', 'harry', 'trump', 'donald', 'truth', 'sojourner', 'tucker',
                'gideon', 'tudor', 'john', 'tugia', 'manasa', 'tuibua', 'esala', 'turner', 'ted', 'turtledove', 'harry',
                'tutu', 'desmond', 'twain', 'mark', 'tweed', 'william', 'marcy', 'tyger', 'frank', 'tynan', 'kenneth',
                'aaron', 'hank', 'abagnale', 'frank', 'abbey', 'edward', 'abel', 'reuben', 'abelson', 'hal', 'abourezk',
                'james', 'abrams', 'creighton', 'ace', 'jane', 'acton', 'john', 'adams', 'abigail', 'adams', 'douglas',
                'adams', 'henry', 'adams', 'john', 'adams', 'john', 'quincy', 'adams', 'samuel', 'adams', 'scott',
                'addams', 'jane', 'addison', 'joseph', 'adorno', 'theodor', 'adler', 'alfred', 'aeschylus', 'aesop',
                'affleck', 'ben', 'agena', 'keiko', 'agnew', 'spiro', 'ahbez', 'eden', 'ahern', 'bertie', 'ah', 'koy',
                'james', 'ahmad', 'aiken', 'clay', 'aiken', 'conrad', 'akinola', 'peter', 'jasper', 'al at', 'alba',
                'jessica', 'alberti', 'leone', 'battista', 'albom', 'mitch', 'alcott', 'louisa', 'may', 'alcuin',
                'aldiss', 'brian', 'alexander', 'the', 'great', 'alexie', 'sherman', 'al hallaj', 'al sadr', 'muqtada',
                'alsahaf', 'muhammed', 'saeed', 'alfven', 'hannes', 'ali', 'ibn', 'abi', 'talib', 'aldrin', 'buzz',
                'ali', 'muhammad', 'ali', 'tariq', 'allee', 'allen', 'agnes', 'allen', 'fred', 'allen', 'james',
                'allen', 'steve', 'allen', 'woody', 'allais', 'alphonse', 'allston', 'aaron', 'amaro', 'rolim', 'amiel',
                'barbara', 'amis', 'martin', 'amos', 'tori', 'andersen', 'hans', 'christian', 'anderson', 'beth',
                'anderson', 'sparky', 'andrians', 'aiven', 'andric', 'ivo', 'angell', 'norman', 'angelou', 'maya',
                'anne', 'princess', 'royal', 'of', 'the', 'united', 'kingdom', 'anthony', 'piers', 'anthony', 'susan',
                'antoniou', 'laura', 'antunes', 'antnio', 'lobo', 'apple', 'fiona', 'aquinas', 'thomas', 'arafat',
                'yasser', 'araya', 'tom', 'arbuthnot', 'john', 'archimedes', 'aristotle', 'armstrong', 'edwin',
                'armstrong', 'louis', 'armstrong', 'neil', 'arp', 'hans', 'arslan', 'alp', 'ascham', 'roger', 'ashe',
                'arthur', 'ashlag', 'baruch', 'ashlag', 'yehuda', 'asimov', 'isaac', 'attali', 'jacques', 'atatrk',
                'mustafa', 'kemal', 'atwood', 'margaret', 'au', 'auden', 'wystan', 'hugh', 'audette', 'derek',
                'augustine', 'of', 'hippo', 'augustine', 'norman', 'ralph', 'austen', 'jane', 'austin', 'alfred',
                'austin', 'austin', 'stone', 'cold', 'steve', 'avenue', 'ba', 'ba', 'jin', 'baba', 'meher', 'baba',
                'tupeni', 'babbage', 'charles', 'babbitt', 'milton', 'bacevich', 'andrew', 'bach', 'richard',
                'bachelard', 'gaston', 'bachelot', 'roselyne', 'bacon', 'francis', 'baddiel', 'david', 'baden powell',
                'sir', 'robert', 'badiou', 'alain', 'badnarik', 'michael', 'baez', 'joan', 'bagehot', 'walter',
                "baha'u'llah", 'bailey', 'philip', 'james', 'baillie', 'bruce', 'bainimarama', 'frank', 'baker', 'jack',
                'baker', 'russell', 'bakhtiari', 'marjaney', 'bakunin', 'mikhail', 'ball', 'hugo', 'ballmer', 'steve',
                'balzac', 'honor', 'de', 'bancroft', 'anne', 'bangs', 'lester', 'banhart', 'devendra', 'banks,ernie',
                'banks', 'robert', 'banks', 'tony', 'barbauld', 'anna', 'letitia', 'barclay', 'william', 'barker',
                'clive', 'barkley', 'charles', 'barlow', 'john', 'perry', 'barnes', 'jack', 'barnes', 'julian',
                'barrie', 'barry', 'dave', 'barry', 'marion', 'bartholin', 'thomas', 'v.', 'baruch', 'bernard',
                'barwich', 'heinz', 'bastiat', 'frdric', 'batch', 'charlie', 'bathgate', 'andy', 'baudelaire',
                'charles', 'baum', 'frank', 'bavadra', 'timoci', 'bayly', 'thomas', 'haynes', 'bazin', 'andr', 'be bl',
                'bean', 'roy', 'beard', 'charles', 'beaumont', 'and', 'fletcher', 'beck', 'glenn', 'becker', 'carl',
                'beckett', 'samuel', 'beddoes', 'mick', 'beecher', 'henry', 'ward', 'beethoven', 'ludwig', 'van',
                'begin', 'menachem', 'bell', 'alexander', 'graham', 'belloc', 'hilaire', 'bellow', 'saul', 'benchley',
                'robert', 'benedict', 'xvi', 'benenson', 'peter', 'ben gurion', 'david', 'benjamin', 'walter', 'benn',
                'tony', 'bennett', 'william', 'andrew', 'cicil', 'bennington', 'chester', 'benson', 'leana', 'bent',
                'silas', 'bentsen', 'lloyd', 'berard', 'edward', 'v.', 'berger', 'ric', 'bergman', 'george', 'bergman',
                'ingmar', 'berio', 'luciano', 'bergerac', 'cyrano', 'de', 'berle', 'milton', 'berlin', 'irving',
                'berne', 'eric', 'bernhard', 'sandra', 'berra', 'yogi', 'berry', 'halle', 'berry', 'wendell', 'bethea',
                'erin', 'bevan', 'aneurin', 'bevel', 'ken', 'bibesco', 'princess', 'elizabeth', 'biden', 'joseph',
                'bierce', 'ambrose', 'biko', 'steve', 'billings', 'josh', 'biondo', 'frank', 'birrell', 'augustine',
                'black', 'elk', 'blair', 'robert', 'blair', 'tony', 'blake', 'william', 'blakey', 'art', 'blalock',
                'jolene', 'blanc', 'mel', 'blanc', 'raymond', 'blanchett', 'cate', 'blix', 'hans', 'blood', 'rebecca',
                'bo', 'boethius', 'ancius', 'bogart', 'neil', 'bohm', 'david', 'bohr', 'niels', 'boileau despreaux',
                'nicholas', 'bojaxhi', 'agnes', 'gonxha', 'bokini', 'ratu', 'ovini', 'bolano', 'roberto', 'bolvar',
                'simn', 'bombeck', 'erma', 'bonaparte', 'napoleon', 'bonhoeffer', 'dietrich', 'bonner', 'elena', 'bono',
                'boone', 'louis', 'boorstin', 'daniel', 'booth', 'william', 'boretz', 'benjamin', 'borges', 'jorge',
                'luis', 'borlaug', 'norman', 'bose', 'subhash', 'chandra', 'bossidy', 'john', 'collins', 'boswell',
                'james', 'botha', 'pik', 'boutroux', 'pierre', 'bowen', 'elizabeth', 'bowerman', 'bill', 'bowie',
                'david', 'bowles', 'chester', 'bowles', 'paul', 'bowles', 'ralston', 'box', 'george', 'br', 'bracken',
                'peg', 'bracken', 'thomas', 'brackett', 'joseph', 'bradbury', 'ray', 'bradley', 'omar', 'braine',
                'john', 'brand', 'max', 'branden', 'nathaniel', 'brando', 'marlon', 'braque', 'georges', 'braun',
                'carol', 'moseley', 'braun', 'wernher', 'von', 'brutigam', 'deborah', 'brautigan', 'richard',
                'brazauskas', 'algirdas', 'brecht', 'bertolt', 'brennus', 'briggs', 'joe', 'bob', 'brilliant',
                'ashleigh', 'brodsky', 'joseph', 'bront', 'emily', 'brooks', 'gwendolyn', 'brooks', 'mel', 'brothers',
                'dr.', 'joyce', 'broun', 'heywood', 'brown', 'alton', 'brown', 'earle', 'brown', 'elizabeth', 'brown',
                'julie', 'brown', 'sam', 'browne', 'harry', 'browne', 'sir', 'thomas', 'browning', 'elizabeth',
                'barrett', 'browning', 'robert', 'broyard', 'anatole', 'bruce', 'lenny', 'bruno', 'giordano', 'brutus',
                'marcus', 'junius', 'bryan', 'william', 'jennings', 'bryant', 'william', 'cullen', 'bryson', 'bill',
                'bu by', 'buber', 'martin', 'buchan', 'john', 'buck', 'pearl', 'buckles', 'frank', 'buddha', 'gautama',
                'buffett', 'warren', 'bujold', 'lois', 'mcmaster', 'bullock', 'sandra', 'bune', 'poseci', 'buuel',
                'luis', 'buonarroti', 'michelangelo', 'burke', 'edmund', 'burnett', 'carol', 'burnham', 'daniel',
                'burnham', 'frederick', 'russell', 'burns', 'edward', 'burns', 'robert', 'burroughs', 'john',
                'burroughs', 'william', 'burton', 'sir', 'richard', 'francis', 'buscaglia', 'leo', 'bush', 'george',
                'bush', 'george', 'bush', 'john', 'carder', 'bush', 'kate', 'bush', 'vannevar', 'butler', 'amir',
                'butler', 'samuel', 'butler', 'samuel', 'butler', 'smedley', 'butler', 'sloss', 'dame', 'elizabeth',
                'buzan', 'tony', 'byrd', 'richard', 'byrd', 'robert', 'byrne', 'david', 'byrne', 'robert', 'byron',
                'lord', 'ca ce', 'cabell', 'james', 'branch', 'caesar', 'irving', 'caesar', 'julius', 'cage', 'john',
                'cain', 'peter', 'callaghan', 'james', 'calvin', 'john', 'cameron', 'julia', 'cameron', 'kirk',
                'campbell', 'beatrice', 'stella', 'camus', 'albert', 'cannon', 'james', 'canseco', 'jos', 'cantona',
                'eric', 'cao,cao', 'apek', 'karel', 'capote', 'truman', 'capone', 'al', 'card', 'orson', 'scott',
                'carducci', 'giosue', 'carey', 'mariah', 'carey', 'sandra', 'carlin', 'george', 'carlson', 'tucker',
                'carlyle', 'thomas', 'carnegie', 'andrew', 'carolla', 'adam', 'carrel', 'alexis', 'carrey', 'jim',
                'carroll', 'lewis', 'carroll', 'tom', 'carson', 'rachel', 'carter', 'elliott', 'carter', 'howard',
                'carter', 'jimmy', 'carville', 'james', 'casals', 'pablo', 'casanova', 'giacomo', 'cash', 'johnny',
                'castaneda', 'carlos', 'castro', 'fidel', 'cather', 'willa', 'cato', 'the', 'elder', 'catt', 'michael',
                'catullus', 'gaius', 'valerius', 'caucau', 'adi', 'asenaca', 'cavell', 'edith', 'cavett', 'dick',
                'cecil', 'robert', 'cervantes', 'miguel', 'de', 'ch ci', 'chambers', 'oswald', 'chamfort', 'chanakya',
                'chandler', 'raymond', 'lon', 'chaney', 'sr.', 'channing', 'william', 'ellery', 'chapin', 'harry',
                'chaplin', 'charlie', 'chapman', 'colin', 'chappelle', 'dave', 'charles', 'ii', 'king', 'of', 'england',
                'charles', 'v', 'holy', 'roman', 'emperor', 'charles', 'ray', 'chateaubriand', 'franois ren', 'de',
                'chatwin', 'bruce', 'chaucer', 'geoffrey', 'chaudhry', 'mahendra', 'chavez', 'cesar', 'cheever', 'john',
                'cherryh', 'chesterton', 'gilbert', 'keith', 'chevalier', 'maurice', 'chiariglione', 'leonardo',
                'chicago', 'judy', 'child', 'julia', 'cho', 'margaret', 'choate', 'rufus', 'christie', 'agatha',
                'chrysostom', 'john', 'chuang', 'chou', 'churchill', 'sarah', 'churchill', 'winston', 'ciccone',
                'madonna', 'cicero', 'cioran', 'emile', 'cl cu', 'clancy', 'tom', 'clapton', 'eric', 'clark', 'frank',
                'clark', 'ramsey', 'clark', 'wesley', 'clarke', 'arthur', 'clausewitz', 'karl', 'von', 'clay', 'andrew',
                'dice', 'clay', 'henry', 'clemenceau', 'georges', 'clinton', 'bill', 'clinton', 'hillary', 'clough',
                'brian', 'cobain', 'kurt', 'donald', 'cochran', 'johnnie', 'cocteau', 'jean', 'codrescu', 'andrei',
                'cohen', 'catman', 'cohen', 'leonard', 'cohen', 'nick', 'cohen', 'richard', 'cole', 'nat', 'coleridge',
                'samuel', 'taylor', 'collett', 'camilla', 'colette', 'collier', 'jeremy', 'collins', 'tim', 'collison',
                'chris', 'colson', 'charles', 'colten', 'james', 'colton', 'charles', 'caleb', 'coltrane', 'john',
                'columbus', 'christopher', 'commager', 'henry', 'steele', 'confucius', 'congreve', 'william', 'conlon',
                'fred', 'connolly', 'cyril', 'conrad', 'joseph', 'conway', 'anne', 'conway', 'morris', 'simon', 'cook',
                'peter', 'cooley', 'mason', 'coolidge', 'calvin', 'cooper', 'alice', 'cooper', 'diana', 'copeland',
                'stewart', 'copernicus', 'nicolaus', 'copland', 'aaron', 'corea', 'chick', 'corea', 'vernon', 'corey',
                'peter', 'corgan', 'billy', 'cornstalk', 'cortzar', 'julio', 'cosby', 'bill', 'coulter', 'ann',
                'coupland', 'douglas', 'courtney', 'leonard', 'h', 'courtney', 'margaret', 'covington', 'stephen',
                'coward', 'noel', 'cowley', 'abraham', 'cowper', 'william', 'crace', 'jim', 'craik', 'dinah', 'crane',
                'frank', 'crane', 'stephen', 'cray', 'seym']

excluded_tags = ['JJR', 'JJS', 'RBR', 'RBS', 'DT', 'CC', 'CD', 'FW', 'MD', 'PRP', 'PRP$', 'RP', 'TO', 'UH', 'WDT', 'WP',
                 'WP$', 'WRB', 'PDT', 'POS', 'LS', 'EX', 'NNP', 'NNPS', 'RBR', 'RBS', 'JJR', 'JJS', 'IN']

verb_list =  ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
nouns_list = ['NN','NNS']

# timer
def hms_string(sec_elapsed):
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60
    return "{}:{:>02}:{:>05.2f}".format(h, m, s)


# save and read vectors and word counts as pickle objects.
def counts_save_to_file(word_c, file_name):
    with open(file_name, 'wb') as handle:
        pickle.dump(word_c, handle)
def vectors_save_to_file(word_v, file_name):
    with open(file_name, 'wb') as handle:
        pickle.dump(word_v, handle)
def counts_read_from_file(file_name):
    with open(file_name, 'rb') as handle:
        count_pickle = pickle.load(handle)
    return count_pickle
def vectors_read_from_file(file_name):
    with open(file_name, 'rb') as handle:
        vector_pickle = pickle.load(handle)
    return vector_pickle

# save and read vectors and word counts as csv files.
def counts_save_to_file_f(word_c, file_name):
    with open(file_name, 'w') as f:
        for key, value in word_c.items():
            f.write('%s:%s\n' % (key, value))
def vectors_save_to_file_f(word_v, file_name):
    with open(file_name, 'w') as f:
        for key, value in word_v.items():
            f.write('%s:%s\n' % (key, value))
def vectors_read_from_file_f(file_name):
    data = {}
    word_vectors = {}
    with open(file_name) as raw_data:
        for item in raw_data:
            if ':' in item:
                key, value = item.split(':', 1)
                data[key] = value
            else:
                pass  # deal with bad lines of text here
    for i, j in list(data.items()):
        w1_i = re.sub(r"\'|\(|\)", "", i)
        w1_j = re.sub(r"\'|\(|\)", "", j).strip('\n')
        word_vectors[w1_i] = w1_j
    return word_vectors
def counts_read_from_file_f(file_name):
    data = {}
    with open(file_name) as raw_data:
        for item in raw_data:
            if ':' in item:
                key, value = item.split(':', 1)
                data[key] = value
            else:   # deal with bad lines of text here
                pass
    for x, y in list(data.items()):
        data[x] = int(y.strip('\n'))

    return data


######################################################################################################################
######################################################################################################################
######################################################################################################################

# Stage 1 data cleaning, word and vectors generation

# remove brackets from word
def strip_tag_name(t):
    idx = t.rfind("}")
    if idx != -1:
        t = t[idx + 1:]
    return t
# read the wikipedia file and create the word vectors
def read_files_and_create_vectors(file_name):
    word_vectors_nouns = {}
    word_vectors_verbs = {}
    word_vectors_adj = {}
    word_vectors_adv = {}
    totalCount = 0  # number of main xml elements
    articles = 0  # number of articles
    window = 5  # how many words before and after are used for creating word vectors / 10[token window
    word_count = {}  # number of occurrences of every word
    flag_wv = False
    # parse the xml file
    for event, elem in etree.iterparse(file_name, events=('start', 'end')):
        tname = strip_tag_name(elem.tag)

        # find tag
        if event == 'start':
            if tname == 'page':
                title = ''
                text = ''
                id = -1
                redirect = ''
                inrevision = False
                ns = 0
            elif tname == 'revision':
                # Do not pick up on revision id's
                inrevision = True
        else:
            if tname == 'title':
                title = elem.text
            elif tname == 'id' and not inrevision:
                id = int(elem.text)
            elif tname == 'redirect':
                redirect = elem.attrib['title']
            elif tname == 'ns':
                ns = int(elem.text)
            elif tname == 'text':
                text = elem.text
            elif tname == 'page':
                totalCount += 1

                # real article found
                if ns != 10 and len(redirect) == 0:
                    articles += 1

                    # print(text)
                    # split on sentences
                    if text is None: continue
                    if not text:
                        continue
                    sentences = tokenize.sent_tokenize(text)
                    for s in sentences:
                        # clean text
                        wiki2plain = Wiki2Plain(s)
                        s = wiki2plain.text
                        # words tonenization / tagging
                        words = TextBlob(s)
                        words = words.tags
                        # filter words greater than 2 characters and less than 23 characters.
                        words = [(word.lower(), tag) for word, tag in words if len(word) > 2 and len(word) < 23]
                        # filter words within the excluded tags.
                        for word in words:
                            if word[1] in excluded_tags:
                                continue
                        # filter stop words, basic words, countries list, cities list, person names and others.
                        words_list = [stop_words_basic_words, countries_list, cities_list, others, person_names]
                        for ls in words_list:
                            words = [word for word in words if not word[0] in ls]
                        # filter out roman characters.
                        for rw in special_list:
                            words = [word for word in words if rw not in word[0]]
                        # filter none words.
                        words = list(filter(None, words))
                        words = list(filter(lambda word: filter(str.isalnum, word), words))
                        if not words: continue
                        # iterate over all words in corpus
                        for i, w in enumerate(words):
                            # count  words
                            if w[0] not in word_count:
                                word_count[w[0]] = 1
                            else:
                                word_count[w[0]] += 1
                            # generate vector for each word / 10 token window
                            min = 0 if 0 > i - window - 1 else i - window  # start index
                            max = len(words) if len(words) - 1 < i + window + 1 else i + window + 1  # end index
                            neigh_words = words[min:i] + words[
                                                         i + 1:max]  # get words before and after ( 10-token window)
                            neigh_words = [i[0] for i in neigh_words]
                            cont = Counter(neigh_words)  # create counter of them
                            if not cont: continue
                            # craete vectors for nouns, verbs, adjectives and adverbs.
                            if w[1] in nouns_list:
                                if w in word_vectors_nouns:
                                    word_vectors_nouns[w] = word_vectors_nouns[w] + cont
                                else:
                                    word_vectors_nouns[w] = cont
                            elif w[1] in verb_list:
                                if w in word_vectors_verbs:
                                    word_vectors_verbs[w] = word_vectors_verbs[w] + cont
                                else:
                                    word_vectors_verbs[w] = cont
                            elif w[1] == 'JJ':
                                if w in word_vectors_adj:
                                    word_vectors_adj[w] = word_vectors_adj[w] + cont
                                else:
                                    word_vectors_adj[w] = cont
                            elif w[1] == 'RB':

                                if w in word_vectors_adv:
                                    word_vectors_adv[w] = word_vectors_adv[w] + cont
                                else:
                                    word_vectors_adv[w] = cont

                # stop after a certain number of articles.
                if articles > 30000:
                    break
                if totalCount > 1 and (totalCount % 500) == 0:
                    print("{:,}".format(totalCount))
                if totalCount > 1 and (totalCount % 1000) == 0:
                    print("{:,}".format(totalCount))
            elem.clear()

    return (word_count, word_vectors_nouns, word_vectors_verbs, word_vectors_adj, word_vectors_adv)
#remove words with less than 3 occurences.
def remove_words(word_c, word_vec_n,word_vec_v,word_vec_adj,word_vec_adv):
    # remove words with less than 3 occurences from the word counts and vector header.
    for word, count in list(word_c.items()):
        if count < 3:
            del word_c[word]
            for item in list(word_vec_n.keys()):
                if word in item:
                    word_vec_n.pop(item)
                    if item not in word_vec_n.keys():
                        break

            for item in list(word_vec_v.keys()):
                if word in item:
                    word_vec_v.pop(item)
                    if item not in word_vec_v.keys():
                        break

            for item in list(word_vec_adj.keys()):
                if word in item:
                    word_vec_adj.pop(item)
                    if item not in word_vec_adj.keys():
                        break

            for item in list(word_vec_adv.keys()):
                if word in item:
                    word_vec_adv.pop(item)
                    if item not in word_vec_adv.keys():
                        break
    # remove words with < 3 occurences that exists in the 10 token window for each word.
    for k, v in list(word_vec_n.items()):
        for o, p in list(v.items()):
            if p < 3:
                v.pop(o)
    for k, v in list(word_vec_v.items()):
        for o, p in list(v.items()):
            if p < 3:
                v.pop(o)
    for k, v in list(word_vec_adj.items()):
        for o, p in list(v.items()):
            if p < 3:
                v.pop(o)

    for k, v in list(word_vec_adv.items()):
        for o, p in list(v.items()):
            if p < 3:
                v.pop(o)

        # remove empty list
    for i, j in list(word_vec_n.items()):
        if j == {}:
            word_vec_n.pop(i)

    for i, j in list(word_vec_v.items()):
        if j == {}:
            word_vec_v.pop(i)

    for i, j in list(word_vec_adj.items()):
        if j == {}:
            word_vec_adj.pop(i)

    for i, j in list(word_vec_adv.items()):
        if j == {}:
            word_vec_adv.pop(i)
    # return the word count (corpus) and the word vectors
    return word_c, word_vec_n,word_vec_v,word_vec_adj,word_vec_ad
# compute complexity as defined in the paper
def final_complexity(word, word_count, word_count_simple):
    if word not in word_count.keys():
        return None
    if word not in word_count_simple.keys():
        return None
    cw = word_count[word] / word_count_simple[word]
    lw = len(word)
    return cw * lw


#Process and steps to run the code of stage 1 (simple and English Wiki):

#Step 1 / Simple Wiki
#######################################

# generate word vectors and word counts for simple wiki.
(word_c_s, word_vec_n_s,word_vec_v_s,word_vec_adj_s,word_vec_adv_s) = read_files_and_create_vectors('C:/Users/together/Desktop/text_simp_project/simplewiki-20180601-pages-articles.xml')

#saving simple vectors, counts to pickle files
counts_save_to_file(word_c_s, "C:/Users/together/Desktop/text_simp_project0/stanford/test30000/read_vec_words/read_vec_simple/before_clean/word_count_simple_1_new_s.pickle")
vectors_save_to_file(word_vec_n_s, "C:/Users/together/Desktop/text_simp_project0/stanford/test30000/read_vec_words/read_vec_simple/before_clean/word_vectors_noun_1_s.pickle")
vectors_save_to_file(word_vec_v_s, "C:/Users/together/Desktop/text_simp_project0/stanford/test30000/read_vec_words/read_vec_simple/before_clean/word_vectors_verb_1_s.pickle")
vectors_save_to_file(word_vec_adj_s, "C:/Users/together/Desktop/text_simp_project0/stanford//test30000/read_vec_words/read_vec_simple/before_clean/word_vectors_adj_1_s.pickle")
vectors_save_to_file(word_vec_adv_s, "C:/Users/together/Desktop/text_simp_project0/stanford/test30000/read_vec_words/read_vec_simple/before_clean/word_vectors_adv_1_s.pickle")

#Step 2 / Simple Wiki
######################################

#reading word counts and word vectors files
word_c_s_ = counts_read_from_file("C:/Users/together/Desktop/text_simp_project0/stanford/test30000/read_vec_words/read_vec_simple/before_clean/word_count_simple_1_new_s.pickle")
word_vec_n_s_ = vectors_read_from_file( "C:/Users/together/Desktop/text_simp_project0/stanford/test30000/read_vec_words/read_vec_simple/before_clean/word_vectors_noun_1_s.pickle")
word_vec_v_s_  = vectors_read_from_file("C:/Users/together/Desktop/text_simp_project0/stanford/test30000/read_vec_words/read_vec_simple/before_clean/word_vectors_verb_1_s.pickle")
word_vec_adj_s_  = vectors_read_from_file("C:/Users/together/Desktop/text_simp_project0/stanford/test30000/read_vec_words/read_vec_simple/before_clean/word_vectors_adj_1_s.pickle")
word_vec_adv_s_ = vectors_read_from_file( "C:/Users/together/Desktop/text_simp_project0/stanford/test30000/read_vec_words/read_vec_simple/before_clean/word_vectors_adv_1_s.pickle")

# Step 3 / Simple Wiki
######################################

# cleaning words with < 3 occurences
(word_count_simple, word_vectors_nouns_s,word_vectors_verbs_s,word_vectors_adj_s,word_vectors_adv_s) = remove_words(word_c_s_, word_vec_n_s_,word_vec_v_s_,word_vec_adj_s_,word_vec_adv_s_)

# save the word counts and vectors in excel sheets
counts_save_to_file_f(word_count_simple, "C:/Users/together/Desktop/text_simp_project0/stanford/test30000/read_vec_words/read_vec_simple/after_clean/word_count_simple_1_new_s.csv")
vectors_save_to_file_f(word_vectors_nouns_s, "C:/Users/together/Desktop/text_simp_project0/stanford/test30000/read_vec_words/read_vec_simple/after_clean/word_vectors_noun_1_s.csv")
vectors_save_to_file_f(word_vectors_verbs_s, "C:/Users/together/Desktop/text_simp_project0/stanford/test30000/read_vec_words/read_vec_simple/after_clean/word_vectors_verb_1_s.csv")
vectors_save_to_file_f(word_vectors_adj_s, "C:/Users/together/Desktop/text_simp_project0/stanford/test30000/read_vec_words/read_vec_simple/after_clean/word_vectors_adj_1_s.csv")
vectors_save_to_file_f(word_vectors_adv_s, "C:/Users/together/Desktop/text_simp_project0/stanford/test30000/read_vec_words/read_vec_simple/after_clean/word_vectors_adv_1_s.csv")


# Step4 / English Wiki
#######################################

# generate word vectors and word counts for English wiki.
(word_c_en, word_vec_n_en,word_vec_v_en,word_vec_adj_en,word_vec_adv_en) = read_files_and_create_vectors('C:/Users/together/Desktop/text_simp_project/enwiki-20180601-pages-articles.xml')

#saving English vectors, counts to pickle files
counts_save_to_file(word_c_en, "C:/Users/together/Desktop/text_simp_project0/stanford/test30000/read_vec_words/read_vec_en/before_clean/word_count_en.pickle")
vectors_save_to_file(word_vec_n_en, "C:/Users/together/Desktop/text_simp_project0/stanford/test30000/read_vec_words/read_vec_en/before_clean/word_vectors_noun_en.pickle")
vectors_save_to_file(word_vec_v_en, "C:/Users/together/Desktop/text_simp_project0/stanford/test30000/read_vec_words/read_vec_en/before_clean/word_vectors_verb_en.pickle")
vectors_save_to_file(word_vec_adj_en, "C:/Users/together/Desktop/text_simp_project0/stanford//test30000/read_vec_words/read_vec_en/before_clean/word_vectors_adj_en.pickle")
vectors_save_to_file(word_vec_adv_en, "C:/Users/together/Desktop/text_simp_project0/stanford/test30000/read_vec_words/read_vec_en/before_clean/word_vectors_adv_en.pickle")

#Step5 /English Wiki
######################################

#reading word counts and word vectors files
word_c_en_ = counts_read_from_file("C:/Users/together/Desktop/text_simp_project0/stanford/test30000/read_vec_words/read_vec_en/before_clean/word_count_en.pickle")
word_vec_n_en_ = vectors_read_from_file( "C:/Users/together/Desktop/text_simp_project0/stanford/test30000/read_vec_words/read_vec_en/before_clean/word_vectors_noun_en.pickle")
word_vec_v_en_  = vectors_read_from_file("C:/Users/together/Desktop/text_simp_project0/stanford/test30000/read_vec_words/read_vec_en/before_clean/word_vectors_verb_en.pickle")
word_vec_adj_en_  = vectors_read_from_file("C:/Users/together/Desktop/text_simp_project0/stanford/test30000/read_vec_words/read_vec_en/before_clean/word_vectors_adj_en.pickle")
word_vec_adv_en_ = vectors_read_from_file( "C:/Users/together/Desktop/text_simp_project0/stanford/test30000/read_vec_words/read_vec_en/before_clean/word_vectors_adv_en.pickle")

#Step6 / English Wiki
######################################

# cleaning words with < 3 occurences
(word_count_en, word_vectors_nouns_en,word_vectors_verbs_en,word_vectors_adj_en,word_vectors_adv_en) = remove_words(word_c_en_, word_vec_n_en_,word_vec_v_en_,word_vec_adj_en_,word_vec_adv_en_)

# save the word counts and vectors in excel sheets
counts_save_to_file_f(word_count_en, "C:/Users/together/Desktop/text_simp_project0/stanford/test30000/read_vec_words/read_vec_en/after_clean/word_count_en.csv")
vectors_save_to_file_f(word_vectors_nouns_en, "C:/Users/together/Desktop/text_simp_project0/stanford/test30000/read_vec_words/read_vec_en/after_clean/word_vectors_noun_en.csv")
vectors_save_to_file_f(word_vectors_verbs_en, "C:/Users/together/Desktop/text_simp_project0/stanford/test30000/read_vec_words/read_vec_en/after_clean/word_vectors_verb_en.csv")
vectors_save_to_file_f(word_vectors_adj_en, "C:/Users/together/Desktop/text_simp_project0/stanford/test30000/read_vec_words/read_vec_en/after_clean/word_vectors_adj_en.csv")
vectors_save_to_file_f(word_vectors_adv_en, "C:/Users/together/Desktop/text_simp_project0/stanford/test30000/read_vec_words/read_vec_en/after_clean/word_vectors_adv_en.csv")


######################################################################################################################
######################################################################################################################
######################################################################################################################

# Stage 2: Possible pairs generation

# compute cosine similarity between word vectors
def counter_cosine_similarity(c1, c2):
    terms = set(c1).union(c2)
    dotprod = sum(c1.get(k, 0) * c2.get(k, 0) for k in terms)
    magA = math.sqrt(sum(c1.get(k, 0) ** 2 for k in terms))
    magB = math.sqrt(sum(c2.get(k, 0) ** 2 for k in terms))
    if magA * magB == 0:
        return 0
    return dotprod / (magA * magB)

# find synonyms and hypernyms for complex word
def find_synonyms(word):
    w = Word(word)
    synonyms = w.synonyms(relevance=3, form='common')
    return synonyms
def find_hypernyms(word):
    similarity_table = []
    hypernyms_final = []
    hypernyms_init = []
    for syn in wordnet.synsets(word):
        hypernyms_init.append(syn)

    for i in range(0, len(hypernyms_init)):
        for j in range(i + 1, len(hypernyms_init)):
            flist1 = hypernyms_init[i]
            flist2 = hypernyms_init[j]
            similarity = flist1, flist2, flist1.wup_similarity(flist2)
            similarity_table.append(similarity)

    high = 0
    set = []
    hypernyms = ''
    for x in similarity_table:
        # print(x[2])
        if x[2] == None:
            continue
        if x[2] > high:
            high = x[2]
            set.append(x)
    for se in set:
        if len(set) == 1:
            hypernym = se[0:2]
        elif len(set) > 1:
            hypernym = set[-1][0:2]
        else:
            pass

        for h in hypernym:
            for l in h.lemmas():
                hypernyms = hypernyms + ' ' + l.name()

    parser = Parser()
    final_list = parser.find_keywords(hypernyms)[0:5]
    final_list = [i for i in final_list if i != word and i[0:4] != word[0:4]]

    for num in final_list:
        if num not in hypernyms_final:
            hypernyms_final.append(num)
    return hypernyms_final

# Generating possbile pairs
def calculate_similarity(word_vectors, word_count, word_vectors_simple, word_count_simple):
    lemmatizer = WordNetLemmatizer()
    possible_pairs = {}

    # compare the words pairs / possible pair generation

    # itereates over all the words in english and simple wiki word_vectors
    for i in range(len(word_vectors) - 1):
        w1 = list(word_vectors)[i].split(', ')  # get word w1 from word_vector
        w1_a = w1[0]  # English wiki word.
        tag_w1 = w1[1]  # get tag of English wiki word.

        # check existance of w1 in simple wiki word counts
        if w1_a not in word_count_simple.keys():
            continue
        l1 = lemmatizer.lemmatize(w1_a)  # get lemmas of the word

        # iterates over all the words of simple_wiki_vector
        for j in range(len(word_vectors_simple) - 1):
            w2 = list(word_vectors_simple)[j].split(', ')  # get word vector w2(simple word)
            w2_b = w2[0]  # simple wiki word.
            tag_w2 = w2[1]  # simple wiki word tag.

            # check existance of w2 in English wiki.
            if w2_b not in word_count.keys():
                continue

            # both words w1 and w2 shouldn't share more than 3 starting letters
            if w1_a[0:4] == w2_b[0:4]:
                continue
            #  remove words which share a common lemma
            l2 = lemmatizer.lemmatize(w2_b)  # get lemmas of the word( possible complex word)
            if l1 == l2:
                continue
            # From all remaining word pairs, we select those in which the second word, in its first sense
            # is a synonym or hypernym of the first.
            s1 = find_synonyms(w1_a)
            if w2_b not in s1:
                h1 = find_hypernyms(w1_a)
                if w2_b not in h1:
                    continue

            # compute complexity of w1 and w2
            c1 = final_complexity(w1_a, word_count, word_count_simple)
            c2 = final_complexity(w2_b, word_count, word_count_simple)
            if c1 is None or c2 is None:
                continue
            # if complexity of English wiki word > complexity of simple wiki word.
            if c1 > c2:
                if tag_w1 in nouns_list and tag_w2 in nouns_list:
                    if tag_w1 == 'NN' and tag_w2 == 'NN':
                        if w1_a not in possible_pairs:
                            possible_pairs[w1_a] = {w2_b}
                        else:
                            possible_pairs[w1_a].add(w2_b)
                    elif tag_w1 == 'NNS' and tag_w2 == 'NNS':
                        if w1_a not in possible_pairs:
                            possible_pairs[w1_a] = {w2_b}
                        else:
                            possible_pairs[w1_a].add(w2_b)

                    elif tag_w1 == 'NN' and tag_w2 == 'NNS':
                        singularize(w2_b)
                        if w1_a not in possible_pairs:
                            possible_pairs[w1_a] = {singularize(w2_b)}
                        else:
                            possible_pairs[w1_a].add(singularize(w2_b))
                    elif tag_w1 == 'NNS' and tag_w2 == 'NN':
                        pluralize(w2_b)
                        if w1_a not in possible_pairs:
                            possible_pairs[w1_a] = {pluralize(w2_b)}
                        else:
                            possible_pairs[w1_a].add(pluralize(w2_b))
                elif tag_w1 in verb_list and tag_w2 in verb_list:
                    ver = Verbs()
                    # generate all verbs consistent forms. solve the case when  verb_list_a is None or verb_list_b is None / also if length of verb_list_b (simple verb) is less than verb_list_a.
                    verb_list_a = ver.lexeme(w1_a)
                    verb_list_b = ver.lexeme(w2_b)
                    if len(verb_list_b) < len(verb_list_a):
                        diff_len = len(verb_list_a) - len(verb_list_b)
                        verb_list_a = verb_list_a[:-diff_len]

                    for i in range(0, len(verb_list_a)):
                        if verb_list_a[i] in possible_pairs:

                            possible_pairs[verb_list_a[i]].add(verb_list_b[i])
                        else:
                            possible_pairs[verb_list_a[i]] = {verb_list_b[i]}
                elif tag_w1 == 'JJ' and tag_w2 == 'JJ':
                    if w1_a not in possible_pairs:
                        possible_pairs[w1_a] = {w2_b}
                    else:
                        possible_pairs[w1_a].add(w2_b)
                elif tag_w1 == 'RB' and tag_w2 == 'RB':
                    if w1_a not in possible_pairs:
                        possible_pairs[w1_a] = {w2_b}
                    else:
                        possible_pairs[w1_a].add(w2_b)


    # save possible pairs to excel.
    vectors_save_to_file(noun_pairs, 'C:/Users/together/Desktop/text_simp_project0/vectorssplit/pairs/noun_pairs.csv')
    vectors_save_to_file(verb_pairs, 'C:/Users/together/Desktop/text_simp_project0/vectorssplit/pairs/verb_pairs.csv')
    vectors_save_to_file(adj_pairs, 'C:/Users/together/Desktop/text_simp_project0/vectorssplit/pairs/adj_pairs.csv')
    vectors_save_to_file(adv_pairs, 'C:/Users/together/Desktop/text_simp_project0/vectorssplit/pairs/adv_pairs.csv')

    return possible_pairs

# Step1 / reading word counts and word vectors
##############################################

word_vectors_simple = vectors_read_from_file ("/home/hinnawe/text_simplification/textblob/test30000/word_vectors_simple_30000_textblob_s.pickle")
word_vectors = vectors_read_from_file ("/home/hinnawe/text_simplification/textblob/test30000/word_vectors_30000_textblob_en.pickle")
word_count_simple = counts_read_from_file ("/home/hinnawe/text_simplification/textblob/test30000/word_count_simple_30000_textblob_s.pickle")
word_count = counts_read_from_file("/home/hinnawe/text_simplification/textblob/test30000/word_count_30000_textblob_en.pickle")

# Step2 / Generate noun, verb, adjective and adverb pairs.
##############################################

pairs = calculate_similarity(word_vectors, word_count, word_vectors_simple, word_count_simple)



######################################################################################################################
######################################################################################################################
######################################################################################################################

# Stage 3 : Sentence simplification

# replace complex words with simple words from the list of all possible pairs.
def simplification(file_name, word_vectors, word_vectors_simple, pairs):
    window = 5  # how many words before and after are used for creating word vectors
    totalArticlesCount = 0
    # parse the xml file
    file = open(file_name, 'r', encoding="utf8")
    lines = file.readlines()
    file.close()

    simplified_sentences = []
    for s in lines:
        s = s.lower()

        wiki2plain = Wiki2Plain(s)
        s = wiki2plain.text
        words = tokenize.word_tokenize(s)
        if words is None: continue
        words = [word for word in words if len(word) > 2 and len(word) < 23]
        # remove stop words, basic words, coutry list, cities list, person names and other words.
        words_list = [stop_words_basic_words, countries_list, cities_list, others, person_names]
        for ls in words_list:
            words = [word for word in words if not word in ls]
        # remove words that holds url, html...
        for rw in special_list:
            words = [word for word in words if rw not in word]
        # filter None
        words = list(filter(None, words))
        words = list(filter(lambda word: filter(str.isalnum, word), words))
        # does not attempt to simplify sentences with less than seven content words
        if len(words) < 7:
            continue

        # iterate words , each content word is examined in order
        for i, w in enumerate(words):
            if w not in word_vectors:
                continue
            # for each replacement word in the possible pairs do the following:
            # For each rule w --> x, unless the replacement word x already appears in the sentence
            similarities = dict()
            for x in pairs:
                if x in words:  # avoid comparing same word in pairs and words.
                    continue
                if x not in word_vectors_simple: # replacement word x should be in word vector
                    #  simple to avoid error when calculating similarity
                    continue
                # build vector of sentence context
                min = 0 if 0 > (i - window - 1) else (i - window)  # start index
                maxim = (len(words)) if (len(words) - 1) < (i + window + 1) else (i + window) + 1  # end index
                neigh_words = words[min:i] + words[i + 1:maxim]  # get words before and after
                scv = Counter(neigh_words)  # create scv

                # Calculate the cosine similarity of CV of word and SCV
                sim_cv_scv = counter_cosine_similarity(scv, word_vectors[w])

                # If this value is larger than a manually specified threshold (0.1 in our experiments)
                # do not use this rule
                if sim_cv_scv > 0.1:
                    continue

                # create a common context vector ccv
                ccv = word_vectors[w] & word_vectors_simple[x]

                # calculate the cosine similarity of the common
                # context vector and the sentence context vector
                context_sim = counter_cosine_similarity(ccv, scv)
                similarities[x] = context_sim
                # If the context similarity is larger than a threshold
                # (0.01), we use this rule to simplify
                rep = max(similarities, key=similarities.get)
                if (similarities[rep] > 0.01):
                    print("simplified sentence is: ",simplified_sentences.append(s.replace(w, rep)))

    return simplified_sentences

# Generate the output sentences.
simplified_sent = simplification('input_external.txt' ,word_vectors,word_vectors_simple ,pairs)