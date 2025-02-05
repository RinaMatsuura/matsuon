import streamlit as st
import tempfile
from openai import OpenAI
import os
import numpy as np
import soundfile as sf  # soundfileをインポート

# ページ設定を行う
st.set_page_config(
    page_title="松浦の実験ページ",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded"
)

def split_audio_file(file_path, chunk_length=60):
    """音声ファイルを指定した長さ（秒）で分割する"""
    try:
        # 音声ファイルを読み込む
        data, sample_rate = sf.read(file_path)  # soundfileを使用して音声ファイルを読み込む
        
        # 分割されたファイルのリストを作成
        split_files = []
        total_samples = len(data)
        chunk_samples = chunk_length * sample_rate  # チャンクのサンプル数
        
        for i in range(0, total_samples, chunk_samples):
            chunk = data[i:i + chunk_samples]  # チャンクを取得
            chunk_file_path = f"{file_path}_part{i // sample_rate}.wav"  # 分割ファイルのパス
            sf.write(chunk_file_path, chunk, sample_rate)  # 分割ファイルをエクスポート
            split_files.append(chunk_file_path)  # 分割ファイルのリストに追加
        
        return split_files  # 分割されたファイルのリストを返す
    except Exception as e:
        st.error(f"音声ファイルの分割中にエラーが発生しました: {str(e)}")
        return []

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

            # 音声ファイルを分割
            split_files = split_audio_file(temp_file_path, chunk_length=60)  # 60秒ごとに分割

            # Whisper APIを使用して文字起こし
            st.subheader("🔍 会話の分析")
            all_transcriptions = []  # すべての文字起こし結果を保存するリスト

            for split_file in split_files:
                with open(split_file, "rb") as audio_file:
                    transcription = client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file,
                        language=language_code[language],
                        response_format="verbose_json"
                    )
                    all_transcriptions.append(transcription.text)  # 各ファイルの文字起こし結果をリストに追加

            # すべての文字起こし結果を表示
            for idx, transcription in enumerate(all_transcriptions):
                st.write(f"### 分割ファイル {idx + 1} の文字起こし結果:")
                st.write(transcription)

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
    - ファイルサイズの上限は100MB
    - 対応フォーマット: MP3, M4A, WAV
    - 音声は明瞭なものを使用することで精度が向上します
    """)

# フッター
st.sidebar.markdown("---")
st.sidebar.write("バージョン: 1.0.0")
st.sidebar.write("© 2024 まつりな")
