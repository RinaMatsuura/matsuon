import streamlit as st
import tempfile
from openai import OpenAI
import os
import subprocess

# ページ設定を行う
st.set_page_config(
    page_title="松浦の実験ページ",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded"
)

def check_audio_format(file_path):
    """音声ファイルの形式をチェックし、必要に応じて変換する"""
    try:
        # ffprobeでファイル情報を取得
        cmd = ['ffprobe', '-i', file_path, '-show_entries', 'format=format_name', '-v', 'quiet', '-of', 'csv=p=0']
        format_name = subprocess.check_output(cmd).decode('utf-8').strip()
        
        if format_name not in ['mp3', 'wav', 'm4a']:
            # 変換が必要な場合
            new_path = file_path + '.mp3'
            convert_cmd = ['ffmpeg', '-i', file_path, '-acodec', 'libmp3lame', '-y', new_path]
            subprocess.run(convert_cmd, check=True)
            os.remove(file_path)
            return new_path
        return file_path
    except Exception as e:
        st.error(f"音声ファイルの処理中にエラーが発生しました: {str(e)}")
        return None

st.title("音声文字起こし 🎤")

# ページ内で言語選択
st.subheader("文字起こしの言語を選択")
language = st.selectbox(
    "言語を選択してください",
    ["日本語", "英語", "自動検出"],
    index=0
)

language_code = {
    "日本語": "ja",
    "英語": "en",
    "自動検出": None
}

# ファイルアップローダーの追加
uploaded_file = st.file_uploader("音声ファイルをアップロード", type=['mp3', 'm4a', 'wav'])

if uploaded_file is not None:
    with st.spinner("文字起こしを実行中..."):
        # 一時ファイルとして保存
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_file_path = temp_file.name

        try:
            client = OpenAI()

            # Whisper APIを使用して文字起こし
            with open(temp_file_path, "rb") as audio_file:
                transcription = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language=language_code[language],
                    response_format="verbose_json"
                )

            # GPT-4による要約と整理
            st.subheader("🔍 会話の分析")
            response = client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": """
                    あなたは会話文の整理と文字起こしの専門家です。以下の指示に従って会話を整理してください：

                    ## 必須タスク
                    1. 複数の話者の発言を明確に区別し、整理してください
                    2. 以下のフォーマットで出力してください：

                    ### サマリー
                    - 会話の要約
                    - 次のアクション項目

                    ### 会話ログ
                    スピーカーA：発言内容
                    スピーカーB：発言内容
                    （時系列順に記載）

                    ## 出力形式
                    - 話者の区別は「話者名：」の形式で明示
                    - 時系列順に会話を整理
                    - 箇条書きで見やすく整形
                    """},
                    {"role": "user", "content": f"以下のテキストをまとめてください：\n{transcription.text}"}
                ],
                temperature=0,
                max_tokens=4096,
                top_p=0.1,
                presence_penalty=0,
                frequency_penalty=0
            )

            st.write(response.choices[0].message.content)

        except Exception as e:
            st.error(f"エラーが発生しました: {str(e)}")
        finally:
            # 一時ファイルの削除
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

# 使い方の説明を更新
with st.expander("💡 使い方"):
    st.write("""
    1. ページ内で文字起こしの言語を選択
    2. 音声ファイル（mp3, m4a, wav）をアップロード
    3. 自動で文字起こしが開始されます
    4. GPT-4による会話の分析結果が表示されます
    
    注意事項：
    - ファイルサイズの上限は25MB
    - 対応フォーマット: MP3, M4A, WAV
    - 音声は明瞭なものを使用することで精度が向上します
    """)

# フッター
st.sidebar.markdown("---")
st.sidebar.write("バージョン: 1.0.0")
st.sidebar.write("© 2024 まつりな")
