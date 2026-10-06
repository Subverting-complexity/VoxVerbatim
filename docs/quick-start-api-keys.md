# Quick start: set up your API keys

This guide tells you how to give VoxVerbatim the keys it needs to
transcribe. Do this one time, before your first transcript.

For your first transcript after this, read
[Quick start: transcribe your first file](quick-start-first-transcript.md).

## What an API key is

An API key is a long password for a program. VoxVerbatim sends your
recordings to speech services. Each service uses your key to know that
the request comes from you.

Each service sends the bill to you. VoxVerbatim does not charge you.

**Keep your keys secret.** Anyone with your key can use your account and
spend your money. VoxVerbatim keeps your keys only in your own settings
file. It does not write them into transcripts.

## What you need

You need 2 accounts. The other services are optional.

| Service | Do you need it? | What it does |
| --- | --- | --- |
| ElevenLabs | Yes | Reads each recording. Finds when each word starts and stops. |
| OpenAI | Yes | Reads each recording. Can also decide between words that the services do not agree on. |
| Microsoft MAI | No | Reads each recording as a third opinion. |
| AssemblyAI | No | Listens again to the parts where the services do not agree. |
| Deepgram | No | Not used yet. Ignore it. |

The fastest start is to use only ElevenLabs and OpenAI. This guide shows
you how to do that. You can add the optional services later.

## Step 1: Get your ElevenLabs key

1. Go to <https://elevenlabs.io> and make an account.
2. Open your account page. Find the API keys section under your profile.
3. Make a new key.
4. Copy the key. Paste it into a safe place, such as a password manager.

The site shows the key only one time. If you lose it, make a new key.

## Step 2: Get your OpenAI key

1. Go to <https://platform.openai.com> and make an account.
2. Add a payment method or credit. OpenAI does not accept requests from
   an account with no credit.
3. Open the API keys page.
4. Make a new key.
5. Copy the key. Paste it into a safe place.

The site shows the key only one time. If you lose it, make a new key.

## Step 3: Open the Settings dialog

1. Start VoxVerbatim.
2. Press `Ctrl+,` (Control and comma). Or, select **File**, then
   **Settings...**.

The Settings dialog opens. The focus is on the category list at the
left. Use the `Up` and `Down` arrow keys to move between categories. The
page for the category shows at the right. Press `Tab` to go into the page.

## Step 4: Type the ElevenLabs key

1. In the category list, select **ElevenLabs**.
2. Make sure that **Use ElevenLabs Scribe** is ticked.
3. Go to the **API key** box. Paste your ElevenLabs key.

The key shows as dots. To see the key, tick **Show the key**.

Below the boxes, a sentence tells you the state of the service. When the
key is in place, it says "ElevenLabs Scribe is switched on and is set up."

## Step 5: Type the OpenAI key

You type the same OpenAI key on 2 pages.

1. In the category list, select **OpenAI transcription**.
2. Make sure that **Use OpenAI for transcription** is ticked.
3. Go to the **API key** box. Paste your OpenAI key.
4. In the category list, select **OpenAI adjudication**.
5. Make sure that **Use OpenAI for adjudication** is ticked.
6. Go to the **API key** box. Paste the same OpenAI key.

The second page is for the model that decides between words that the
services do not agree on. This costs a little more for each run.

If you do not want this, clear **Use OpenAI for adjudication**. Then you
do not need a key on that page. The words that the services do not agree
on wait for you to decide. The transcript tells you that adjudication was
not used.

## Step 6: Switch off the services you do not use

VoxVerbatim starts with Microsoft MAI and AssemblyAI switched on. If you
do not have keys for them, switch them off. If you do not, VoxVerbatim
does not start a transcription.

1. In the category list, select **Microsoft MAI**.
2. Clear **Use Microsoft MAI**.
3. In the category list, select **AssemblyAI**.
4. Clear **Use AssemblyAI**.

A service that is switched off is never used, and you do not pay for it.

## Step 7: Check that everything is ready

1. In the category list, select **Transcription**.
2. Read the sentence about what a run needs.

If everything is correct, it says "Everything a run needs is set up."

If not, it tells you what is missing. Do what it says. For example:

- "ElevenLabs Scribe is switched on but is not set up." Do step 4 again.
- "Adjudication is switched on but is not set up." Do step 5 again. Or,
  if you do not want adjudication, clear **Use OpenAI for adjudication**
  on the **OpenAI adjudication** page.

**Note:** The **Transcription** page also has the box **Let a reasoning
model settle what is left**. Adjudication occurs only when this box and
**Use OpenAI for adjudication** are both ticked. Clear either box to stop
it.

## Step 8: Save

Select **OK**.

If a box has a problem, the dialog does not close. A message tells you
which box and what to change. Correct it, then select **OK** again.

To leave without saving, select **Cancel**.

## Optional: add the other services later

More services give better transcripts, but cost more.

**AssemblyAI.** Make an account at <https://www.assemblyai.com>. Copy the
key from your dashboard. In Settings, select **AssemblyAI**, tick
**Use AssemblyAI** and paste the key.

**Microsoft MAI.** This one takes more work. You must make an Azure AI
Foundry resource in your own Azure subscription. Then:

1. In the Azure portal, open the **Keys and Endpoint** page of your
   resource.
2. Copy one of the 2 keys. Either key works.
3. Copy the endpoint. It looks like
   `https://your-resource.cognitiveservices.azure.com`.
4. In Settings, select **Microsoft MAI** and tick **Use Microsoft MAI**.
5. Paste the key into **API key**.
6. Paste the endpoint into **Resource endpoint**. Do not add anything to
   the end of the address.

Microsoft MAI needs both the key and the endpoint. With only one, it
shows as not set up.

## Optional: check the prices

Before each run, VoxVerbatim tells you the approximate cost. It uses the
rates on the **Costs** page in Settings. Services change their prices,
so compare these rates with each service's own price page.

## If you need help

- Every setting has an explanation. Move the focus to a setting, then
  read the **About this setting** panel at the bottom of the dialog.
- Press `F1` in the Settings dialog to read all the explanations
  together.
