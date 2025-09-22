<h1>Red Light, Green Light🐔</h1>
<h2>BCI P300 EEG Survival Game - Surge BrainHack Fall 2025</h2>
<p>A game inspired by Red Light, Green Light but adapted into survival mode:
There is no finish line — just stay alive by not going off screen. Players control their avatar using A/D (player 1) or the Left/Right arrow (player 2). EEG brain data controls the speed of the avatar. The more alpha input the faster the speed. During red lights the avatar goes backwards, while during green lights it goes forward. Cars also pass randomly as hazards — avoid them to survive. The Last player standing wins.</p>
<h2>Demonstrations</h2>
<figure>
<img width="473" height="600" src="https://github.com/user-attachments/assets/b6d5e2fd-bf31-4bde-aa65-d3c6d11e2b1c"><br>
<figcaption>Playing Game connected with EEG</figcaption><br><hr>
</figure>
<figure>
  <img width="800" height="450" alt="Screenshot (58)" src="https://github.com/user-attachments/assets/e6009ba7-42c0-4691-a363-d880f3ea1cfd" />
<img width="800" height="450" alt="Screenshot (60)" src="https://github.com/user-attachments/assets/609b4381-ed26-4c3c-8feb-15b3c9a497f7" />
 <br> <figcaption>Game Interfaces</figcaption>
</figure>

<h2>Set-Up</h2>
<ul>
  <li>Create a virtual environment:
    <pre><code>python -m venv venv</code></pre>

  <p>If on macOS, run the following command:</p>
  <pre><code>source venv/bin/activate</code></pre>
  
  <p>If on Windows (PowerShell), run the following command:</p>
  <pre><code>venv\Scripts\Activate.ps1</code></pre>
  </li>

  <li> Install Dependencies:
  <pre><code>pip install -r requirements.txt</code></pre>
</code></pre>

<br>

  </li>
  <li>Connect a BrainFlow-compatible EEG board</li>
  <li>Run the game:
    <pre><code>python redlight_greenlight.py</pre></code>
  </li>
  <li>Note:
    <ul>
      <li>If no EEG is available the game falls back to randomized test-mode values.</li>
    </ul>
  </li>
  <li>Controls: 
  <ul>
    <li>Player 1 = <code>A</code>/ <code>D</code>,</li> 
    <li>Player 2 = <code>Left</code> / <code>Right</code>.</li>
    <li> Press <code>X</code> to restart and <code>ESC</code> to quit
    .</li>
</ul>
<hr>
<footer>
  <p>By: Paras, Reese, Jack, Yaniv</p>
</footer>
