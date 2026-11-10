package ai.oazy.phone;

import android.app.Activity;
import android.content.Intent;
import android.graphics.Color;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.view.View;
import android.view.Window;
import android.view.WindowManager;
import android.webkit.ConsoleMessage;
import android.webkit.WebChromeClient;
import android.webkit.WebSettings;
import android.webkit.WebView;
import android.webkit.WebViewClient;

import org.json.JSONObject;

/**
 * The whole app is one page (assets/index.html, in the look of the landing page and the lab dashboard) shown in a
 * WebView; Bridge connects it to the engine (a native program that runs the model on the phone's CPU) and to the
 * model files. Nothing leaves the phone, except a model download the user starts.
 */
public class MainActivity extends Activity {
    static final int PICK_MODEL = 7;
    static final int SAVE_MODEL = 8;
    String saving;

    WebView web;
    Bridge bridge;
    boolean pageReady = false;
    String pendingLink = null;

    @Override
    protected void onCreate(Bundle state) {
        super.onCreate(state);
        Window w = getWindow();
        w.setStatusBarColor(Color.rgb(5, 5, 7));
        w.setNavigationBarColor(Color.rgb(5, 5, 7));
        w.setSoftInputMode(WindowManager.LayoutParams.SOFT_INPUT_ADJUST_RESIZE);
        w.getDecorView().setBackgroundColor(Color.rgb(5, 5, 7));

        web = new WebView(this);
        web.setBackgroundColor(Color.rgb(5, 5, 7));
        web.setOverScrollMode(View.OVER_SCROLL_NEVER);
        WebSettings s = web.getSettings();
        s.setJavaScriptEnabled(true);
        s.setDomStorageEnabled(true);
        s.setAllowFileAccess(false);
        s.setAllowContentAccess(false);
        s.setMediaPlaybackRequiresUserGesture(true);
        s.setTextZoom(100);
        bridge = new Bridge(this);
        web.addJavascriptInterface(bridge, "OazyNative");
        web.setWebChromeClient(new WebChromeClient() {
            @Override
            public boolean onConsoleMessage(ConsoleMessage m) {
                android.util.Log.d("oazy-page", m.message() + " (" + m.sourceId() + ":" + m.lineNumber() + ")");
                return true;
            }
        });
        web.setWebViewClient(new WebViewClient() {
            @Override
            public boolean shouldOverrideUrlLoading(WebView v, String url) {
                if (url.startsWith("file:///android_asset/")) return false;
                try { startActivity(new Intent(Intent.ACTION_VIEW, Uri.parse(url))); } catch (Exception e) { /* no browser */ }
                return true;
            }

            @Override
            public void onPageFinished(WebView v, String url) {
                pageReady = true;
                if (pendingLink != null) { sendLink(pendingLink); pendingLink = null; }
            }
        });
        setContentView(web);
        web.loadUrl("file:///android_asset/index.html");
        handleIntent(getIntent());
    }

    @Override
    protected void onNewIntent(Intent intent) {
        super.onNewIntent(intent);
        handleIntent(intent);
    }

    /** oazy://download?url=... (the pod's share page) or a shared link: offer to download that model. */
    void handleIntent(Intent intent) {
        if (intent == null) return;
        String link = null;
        Uri data = intent.getData();
        if (data != null && "oazy".equals(data.getScheme())) link = data.getQueryParameter("url");
        else if (Intent.ACTION_SEND.equals(intent.getAction())) link = intent.getStringExtra(Intent.EXTRA_TEXT);
        if (link == null || !(link.startsWith("http://") || link.startsWith("https://"))) return;
        if (pageReady) sendLink(link); else pendingLink = link;
    }

    void sendLink(String link) {
        try {
            JSONObject o = new JSONObject();
            o.put("type", "link");
            o.put("url", link);
            event(o);
        } catch (Exception e) { /* cannot happen */ }
    }

    /** Hand an event to the page (from any thread). */
    void event(final JSONObject o) {
        final String js = "window.oazy && oazy.event(" + o.toString() + ")";
        runOnUiThread(new Runnable() {
            public void run() { web.evaluateJavascript(js, null); }
        });
    }

    void createDocument(String name) {
        saving = name;
        Intent i = new Intent(Intent.ACTION_CREATE_DOCUMENT);
        i.addCategory(Intent.CATEGORY_OPENABLE);
        i.setType("application/octet-stream");
        i.putExtra(Intent.EXTRA_TITLE, name);
        startActivityForResult(i, SAVE_MODEL);
    }

    void pickModel() {
        Intent i = new Intent(Intent.ACTION_OPEN_DOCUMENT);
        i.addCategory(Intent.CATEGORY_OPENABLE);
        i.setType("*/*");
        startActivityForResult(i, PICK_MODEL);
    }

    @Override
    protected void onActivityResult(int req, int res, Intent data) {
        super.onActivityResult(req, res, data);
        if (req == PICK_MODEL && res == RESULT_OK && data != null && data.getData() != null) bridge.importFrom(data.getData());
        if (req == SAVE_MODEL && res == RESULT_OK && data != null && data.getData() != null && saving != null) bridge.saveTo(data.getData(), saving);
    }

    @Override
    public void onBackPressed() {
        web.evaluateJavascript("window.oazy ? oazy.back() : false", new android.webkit.ValueCallback<String>() {
            public void onReceiveValue(String handled) {
                if (!"true".equals(handled)) MainActivity.super.onBackPressed();
            }
        });
    }

    @Override
    protected void onPause() {
        super.onPause();
        web.evaluateJavascript("window.oazy && oazy.visible(false)", null);
    }

    @Override
    protected void onResume() {
        super.onResume();
        web.evaluateJavascript("window.oazy && oazy.visible(true)", null);
    }

    @Override
    protected void onDestroy() {
        bridge.shutdown();
        super.onDestroy();
    }

    static boolean atLeast(int api) { return Build.VERSION.SDK_INT >= api; }
}
