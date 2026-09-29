/**
 * The Minerva wordmark.
 *
 * Inlined rather than loaded as an image so it can take its colour from
 * whatever renders it. The file draws dark green on a full-bleed cream
 * plate, which on this sidebar would have been a pale slab with a logo
 * in it rather than a logo. The plate is dropped and the letterforms are
 * currentColor.
 *
 * The viewBox is trimmed to the artwork. On its own canvas the wordmark
 * fills 95 percent of the width but only 67 percent of the height, so
 * the untrimmed box reserved half again as much vertical room as the
 * letters use.
 *
 * Generated from minerva.svg at the repository root, which stays as it
 * was supplied.
 */
export function Logo() {
  return (
    <svg className="logo" viewBox="26 60 1062 270" role="img"
         aria-label="Minerva">
      <g fill="currentColor" transform="translate(10,20)">
      {/* M */}
      <rect x="40" y="60" width="8" height="240"/>
      <polygon points="40,60 76,60 138,272 126,300"/>
      <polygon points="122,300 131,300 206,60 197,60"/>
      <rect x="196" y="60" width="34" height="240"/>
      <rect x="26" y="60" width="50" height="7"/>
      <rect x="22" y="293" width="44" height="7"/>
      <rect x="186" y="60" width="56" height="7"/>
      <rect x="182" y="293" width="62" height="7"/>
      {/* i with diamond dot */}
      <rect x="280" y="112" width="34" height="188"/>
      <polygon points="262,110 314,110 314,120 262,118"/>
      <rect x="262" y="293" width="70" height="7"/>
      <polygon points="297,48 318,70 297,92 276,70"/>
      {/* n */}
      <rect x="360" y="112" width="34" height="188"/>
      <polygon points="342,110 394,110 394,120 342,118"/>
      <path d="M392 138 C408 116 432 106 456 108 C478 110 486 128 486 152 L486 300 L452 300 L452 150 C452 136 444 128 430 128 C414 128 402 138 394 154 Z"/>
      <rect x="344" y="293" width="68" height="7"/>
      <rect x="436" y="293" width="68" height="7"/>
      {/* e (sigma-style) */}
      <rect x="530" y="110" width="84" height="8"/>
      <polygon points="608,110 616,110 618,146 610,146"/>
      <polygon points="528,110 564,110 594,203 580,212"/>
      <polygon points="580,198 592,208 540,300 528,300"/>
      <rect x="528" y="292" width="90" height="8"/>
      <polygon points="610,262 618,262 618,300 608,300"/>
      {/* r */}
      <rect x="660" y="112" width="34" height="188"/>
      <polygon points="642,110 694,110 694,120 642,118"/>
      <path d="M692 146 C704 120 724 108 748 108 L754 108 L754 152 L746 152 C742 136 732 130 722 131 C708 133 700 146 694 162 Z"/>
      <rect x="644" y="293" width="72" height="7"/>
      {/* v */}
      <polygon points="772,110 808,110 864,300 848,300"/>
      <polygon points="844,300 852,300 910,110 902,110"/>
      <rect x="756" y="110" width="66" height="7"/>
      <rect x="886" y="110" width="40" height="7"/>
      {/* a (single-story, bowl crossed like the display o) */}
      <path fill-rule="evenodd" d="M1000 108 C955 108 940 150 940 205 C940 260 955 302 1000 302 C1030 302 1040 280 1040 205 C1040 130 1030 108 1000 108 Z M1002 122 C990 122 976 145 976 205 C976 265 990 288 1002 288 C1014 288 1020 265 1020 205 C1020 145 1014 122 1002 122 Z"/>
      <rect x="976" y="201" width="44" height="7"/>
      <rect x="1020" y="112" width="34" height="188"/>
      <rect x="1020" y="110" width="50" height="7"/>
      <rect x="1004" y="293" width="68" height="7"/>
      </g>
    </svg>
  );
}
