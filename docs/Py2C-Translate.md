# Using Nuitka
Nuitka is a powerful tool that translates your Python code into C, then compiles it into a binary using a C compiler like GCC or MSVC. This makes reverse-engineering significantly harder compared to other tools. Nuitka Standard (version) is open source while Nuitka Commercial (version) is required paid subscription. The core version of Nuitka is distributed under the Apache License 2.0 or GNU Affero General Public License v3 (AGPLv3).

1. Install Nuitka:
cmd: python -m pip install Nuitka

2. Create a Standalone Binary:
- Navigate to your script's folder and run:
- cmd: python -m nuitka --onefile your_script.py
- --onefile: Packs everything into a single .exe (Windows) or binary file (Linux/Mac) for easy distribution.
- --standalone: Creates a folder with the executable and all necessary support files.

Setting up a C compiler for Nuitka depends on your operating system. Nuitka is quite smart—on Windows, it can often handle the setup for you automatically.
For Windows users, the easiest way to set up and test your C compiler is to let Nuitka handle it automatically during a test build.

1. The "Hello World" Test

The best way to verify your setup is to compile a simple script. If a compiler is missing, Nuitka will detect it and ask to download one for you.

- Create a file named test.py with this code: python >> print("Nuitka is working!")
- Open Command Prompt or PowerShell in that folder.
- Run the following command: cmd >> python -m nuitka --onefile test.py

2. What to Expect During the First Run

If you don't already have Visual Studio installed, you will see a prompt in your terminal:

- MinGW64 Prompt: Nuitka will ask: "Is it OK to download and put it in '...'?"
	- Action: Type Yes and press Enter. This downloads a specialized C compiler (MinGW64) that Nuitka knows how to use perfectly.
- CCache Prompt: It may also ask to download ccache.exe.
	- Action: Type Yes. This tool significantly speeds up future compilations by caching previous work.

3. Verify the Output

Once the process finishes (it may take a few minutes for the first time), you should see:

- A new file named test.exe in your folder. 
- Run it by typing .\test.exe. If it prints "Nuitka is working!", your C compiler is correctly configured.


4. Recommendation

To create one .exe that works on almost any Windows PC:

- Install Python 3.8 (32-bit).
- Use the MinGW64 compiler (Nuitka will download this for you).
- Compile with: python -m nuitka --onefile --windows-disable-console your_script.py.
	- Hide the Console: If your script has a GUI (like Tkinter) and you don't want a black command prompt window to pop up, add: --windows-disable-console.
	- Add an Icon: To use your own .ico file, add: --windows-icon-from-ico=your_icon.ico.
	- Add Version Info: You can add your company name and version by using --windows-company-name="MyCompany" and --windows-product-version=1.0.0.
	- Forces the "Run as Administrator" prompt: --windows-uac-admin
	- Ensures all your separate .py files are included and protected: --follow-imports






